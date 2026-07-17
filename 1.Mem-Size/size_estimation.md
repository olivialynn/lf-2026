# Memory-size estimation in `hats-import` (`mem_size` thresholding)

How the importer partitions a catalog so that every partition fits a target
**in-memory** size (`byte_pixel_threshold`, e.g. 1 GB), how the per-row size
model works, what it approximates, and the story of the boolean bit-packing
bug fixed on 2026-07-16.

Code lives in two repos:

- `hats-import` — the pipeline: sampling, mapping, histograms, alignment
  (`src/hats_import/catalog/run_import.py`, `map_reduce.py`, `resume_plan.py`)
- `hats` — the size model and the alignment math
  (`src/hats/io/size_estimates.py`, `src/hats/pixel_math/partition_stats.py`)

---

## 1. Why "size" is ambiguous: the same data has three sizes

Catalog data inflates twice on its way from disk to RAM:

| stage | what it is | how to measure | DP2 object data |
|---|---|---|---|
| on-disk | compressed parquet file | `os.stat` | 35 GB (optimized_0) |
| uncompressed | parquet pages after decompression, still encoded | `_metadata` `total_byte_size` | ~55 GB (~1.4–1.5×) |
| **in-memory** | decoded arrow buffers | `pyarrow` `Table.nbytes` | **113.5 GB (~3.2× disk)** |

The second inflation (decoding) happens because parquet's dictionary and
run-length encodings expand into plain arrow buffers when read. The
`byte_pixel_threshold` budget is about what a partition costs **loaded**, so
the estimator targets the last row: `Table.nbytes` is the ground truth, and
any audit of the builds must use it (the cheaper proxies understate RAM by
1.5–3× and make correctly-sized partitions look badly over-split).

## 2. The pipeline

`mem_size` thresholding replaces "max rows per partition" with "max bytes per
partition". Four steps:

**(a) Sample once** — `get_cols_in_input_file` (`map_reduce.py`) reads the
first chunk of the *first* input file and splits columns into:

- **precomputed** — fixed-width types (ints, floats, bools, datetimes), plus
  string/binary columns whose sampled sizes are consistent (max ≤ 2× mean).
  Their per-row cost is summed once into a single constant,
  `precomputed_row_size` (for strings: the mean over the sampled chunk).
- **variable-length** — lists, nested columns, and any string column too
  inconsistent for a constant. These must be measured row by row.

This assumes all input files share a schema and similarly-sized values; the
one sample is shared by every mapping task.

**(b) Map** — `map_to_pixels` reads each input file in chunks, but only the
variable-length columns (plus RA/Dec) — precomputed columns don't need to be
read at all, which is most of the I/O saving. Each row's size is

```
row_size = measured(variable-length columns) + precomputed_row_size
```

and two histograms are accumulated at the mapping HEALPix order: row counts
and summed bytes per pixel (`supplemental_count_histogram`, int64 — per-row
fractions truncate, ≤1 byte/row, negligible).

**(c) Align** — `generate_alignment` (`hats/pixel_math/partition_stats.py`)
computes nested sums of the byte histogram up the pixel tree, then assigns
each sky region the **coarsest** order whose subtree sum is `<= threshold`.
The comparison is inclusive and there is **no built-in headroom** — if
partitions come out consistently below the budget, the estimate is what's off,
not a deliberate safety margin.

**(d) Split & reduce** — unchanged from row-count mode: shards are written per
destination pixel and concatenated into the final partition files.

## 3. The per-row size model

`get_mem_size_per_row` (`hats/io/size_estimates.py`) prices each row as its
slice of the loaded arrow column buffers. Everything is computed from array
metadata (offsets, lengths, types) — no values are ever converted to Python
objects. Per column type:

| type | per-row cost | matches arrow buffer |
|---|---|---|
| fixed-width (int, float, datetime, decimal) | `bit_width / 8` | values buffer |
| **boolean** | **1/8 byte (bit-packed)** | values buffer |
| string / binary | UTF-8 data bytes + one offset entry (4 B; 8 B for `large_`) | data + offsets buffers |
| list of fixed-width | element count × element width + one offset entry | child values + offsets |
| list of variable-width (e.g. `list<string>`) | element count × column-wide mean bytes/element + offset | exact in aggregate, averaged per row |
| struct (nested columns) | sum of its fields' costs | child buffers |
| validity bitmap | +1/8 byte per row/element **when the bitmap buffer is materialized** | validity buffer |
| anything else (dictionary, union, …) | `column.nbytes / num_rows`, uniform | whole column, exactly |

Pandas input is converted column-by-column to arrow first (cheap for numeric
and arrow-backed columns) so both input kinds are measured with one model;
columns arrow can't represent fall back to per-value `sys.getsizeof`.

Two deliberate exclusions:

- **Python object overhead** is not counted — it prices materializing rows as
  Python objects, not loading the data. (An older version of the estimator
  used `sys.getsizeof` everywhere and badly overcounted for exactly this
  reason.)
- **Offsets buffers** are counted as *n* entries instead of the true *n + 1*
  — a few bytes per column chunk, noise at partition scale.

### The bitmap subtlety

Arrow arrays carry an optional validity bitmap (1 bit/row). The model counts
it when the buffer is **materialized**, not when `null_count > 0`: the parquet
reader allocates an all-valid bitmap for nullable columns, and `Table.nbytes`
includes it either way. On DP2 this was worth ~1.3 MB per float light-curve
field per partition (−1.5% overall) despite zero actual nulls.

## 4. The 2026-07-16 bug: booleans priced at 8× their real cost

**Symptom.** Builds with `byte_pixel_threshold = 1_000_000_000` produced
maximum partitions of 809–869 MB — never near 1 GB — and a merge audit with
exact arrow sizes found sibling groups summing to 872–996 MB that "should"
have been kept coarse. Everything clustered at **87–100% of budget**.

**Root cause.** `_fixed_value_width` returned `max(1, bit_width // 8)` =
**1 byte per boolean**, but arrow stores booleans bit-packed at **1 bit**. The
1-byte convention was defensible for a *top-level* bool column materialized in
numpy-backed pandas — but inside `list<bool>` the data stays arrow bit-packed
in every loaded representation. DP2's `objectForcedSource` nested column
carries **14 `list<bool>` flag fields** (`pixelFlags_*`, `psfFlux_flag`, …),
each estimated at 6.9× its true size: +116 MB on a 715 MB partition, a
**+13–16% overestimate** overall. The builder faithfully split every region
whose *estimate* crossed 1 GB, so real sizes capped at ~1 GB / 1.15 ≈ 870 MB —
matching the observed maxima exactly.

**Fix** (in `hats/src/hats/io/size_estimates.py`):

1. booleans (and any bit-packed type) cost `bit_width / 8` everywhere;
2. validity bitmaps are counted when materialized (see above) — without this,
   the fix would overshoot and let partitions land slightly *over* budget;
3. per-row sizes are floats (1/8-byte granularity), summed before rounding.

**Verification** — model total vs `Table.nbytes` on real DP2 partitions:

| partition | nbytes | model (before) | model (after) |
|---|---|---|---|
| optimized_0 `o7/103576` | 715.0 MB | 831.1 MB (1.162×) | 715.0 MB (1.0000×) |
| optimized_0 `o7/103579` | 42.7 MB | 48.2 MB (1.127×) | 42.7 MB (1.0000×) |
| optimized_2 `o5/4526` | 868.4 MB | 986 MB (1.135×) | 868.4 MB (1.0000×) |

After a rebuild, partitions should press right up against the 1 GB budget,
with ~10–15% fewer, fatter partitions.

## 5. Remaining approximations (aggregate accuracy is what matters)

The alignment only ever consumes **sums over thousands of rows**, so per-row
noise cancels; what matters is aggregate bias. Known residuals, all small:

- offsets counted as *n* vs *n + 1* entries: bytes per column chunk;
- `list<string>` per-row averaging: exact in aggregate by construction;
- `precomputed_row_size` sampled from one chunk of one file: biased if the
  first file's strings aren't representative of the whole input;
- bitmap presence measured on *input* chunks predicts the *output* file's
  buffers — same data, but reader/writer allocation quirks can differ;
- histogram int64 truncation: ≤1 byte per row.

## 6. Auditing it yourself

```python
import pyarrow.parquet as pq
from hats.io import size_estimates

t = pq.read_table(partition_path)
model = sum(size_estimates.get_mem_size_per_row(t))
print(model / t.nbytes)   # ~1.000 on a healthy estimator
```

For whole-catalog audits (size distributions, HEALPix merge analysis with
exact arrow sizes), see `partition_stats.py` and
`2.partition_distribution.ipynb` in this directory.

> **Environment note:** `import hats` in the LSST kernel currently resolves to
> a non-editable hats 0.10.0 in `~/.local/site-packages` (old estimator!),
> shadowing the patched `~/hats` checkout, and `hats_import` is not installed.
> Run with `PYTHONPATH=~/hats/src:~/hats-import/src` or reinstall both repos
> editable before rebuilding.

## 7. Appendix: the buffers behind the model

Everything the size model prices comes down to three kinds of arrow buffer.
The **values buffer** is the obvious one — the actual data, packed end to end.
The other two exist to answer questions the values buffer can't: *"is this
slot real?"* (validity bitmap) and *"where does each row start and end?"*
(offsets). Understanding these two makes every line of the §3 table obvious.

### The validity bitmap: "is this slot real?"

A column of numbers is one tight buffer — `[7, 3, 9, 2, …]`, 8 bytes per slot.
Now suppose row 2 has *no value* (null). What goes in its slot? You can't
leave a hole (the buffer is contiguous), and you can't use a magic number like
-999 or NaN — for most types every bit pattern is a legitimate value a user
might really have.

Arrow's answer: keep the values buffer dumb — put *anything* in the null slot
— and store the "is this real?" information separately, one **bit** per row:

```
values:  [ 7 ][ 3 ][ ?? ][ 2 ]      <- 8 bytes each, slot 2 is garbage
bitmap:    1    1    0     1        <- 1 bit each: valid, valid, NULL, valid
```

Reading row 2, arrow checks the bitmap first: the bit is 0, so it reports null
and never looks at the garbage. The whole concept is "a tiny checklist of
which slots are real." It's cheap — 8 rows per byte, so a million-row column
pays 125 KB for full null support regardless of how wide the values are —
which is exactly the 1/8-byte-per-row term in the model.

Two wrinkles that matter for estimation:

1. **It's optional.** A column with no nulls may skip the bitmap entirely
   ("no checklist = everything's real"), so two columns with identical data
   can differ by 1 bit/row depending on whether the bitmap was allocated.
2. **"No nulls" doesn't mean it's absent.** Some code paths — notably the
   parquet reader, for any column *declared nullable* in the schema —
   allocate the bitmap anyway with every bit set to 1. Harmless, but
   `Table.nbytes` counts it. DP2's light-curve fields hold ~10.6 M elements
   per partition, so each all-valid bitmap is ~1.3 MB — the residual 1.5% the
   estimator missed until it learned to count the bitmap **when the buffer
   exists** rather than when `null_count > 0`.

Fun connection to the boolean bug: arrow stores booleans as bits too, with the
same packing. A bool column is essentially *two* bitmaps stacked — one for
true/false, one for valid/null — which is why the fixed estimator prices a
boolean at 1/8 byte, and why the old 1-byte price was 8× too high.

### Offsets: "where does each row start and end?"

Fixed-width data doesn't need bookkeeping — row *i* of a float64 column lives
at bytes `8i..8i+8`, full stop. But strings and lists have different lengths
per row, so arrow packs all the *content* into one flat buffer and adds an
**offsets buffer**: one integer per row boundary, marking where each row's
slice begins. Row *i* owns `values[offsets[i] : offsets[i+1]]`:

```
strings:  ["g", "rr", "", "izy"]

data:      g r r i z y             <- 6 bytes, all rows concatenated
offsets:   0  1  3  3  6           <- n+1 = 5 entries, 4 bytes each
                 ^--^
                 row 2 is empty: offsets[2] == offsets[3]
```

Consequences the model leans on:

- **Per-row cost is data + one offset entry.** Row *i* costs
  `offsets[i+1] - offsets[i]` data bytes plus its 4-byte offset slot (8-byte
  for `large_string` / `large_list`). An empty or null row still costs its
  offset entry — that's why the §3 table adds "+ one offset entry" to every
  string and list row, and why a null light curve isn't free.
- **Measuring is free.** The model never touches the content: subtracting
  neighboring offsets (`pc.binary_length`, `pc.list_value_length`) yields
  every row's size in one vectorized pass. This is what lets the mapper price
  millions of rows per second.
- **Lists nest the same trick.** A `list<float64>` column is an offsets buffer
  over a flat float64 values buffer; a `list<string>` is offsets over offsets
  over bytes. Nested light curves (struct of lists) are just several of these
  stacked side by side, one per field — each with its own offsets, and each a
  candidate for its own validity bitmap.
- **The n + 1 quirk.** *n* rows need *n + 1* fence posts, but the model counts
  one entry per row — undercounting exactly one entry (4–8 bytes) per offsets
  buffer per column chunk. At partition scale this is noise, and it's the
  reason the §5 list says "offsets counted as *n* vs *n + 1*."
