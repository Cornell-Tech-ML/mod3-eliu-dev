# MiniTorch Module 3

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module3.html


You will need to modify `tensor_functions.py` slightly in this assignment.

* Tests:

```
python run_tests.py
```

* Note:

Several of the tests for this assignment will only run if you are on a GPU machine and will not
run on github's test infrastructure. Please follow the instructions to setup up a colab machine
to run these tests.

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py

# Task 3.1 and 3.2
## Parallel Processing Diagnostics

```
MAP

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_map.<locals>._map,
/Users/ericliu/Desktop/Github/cornell/cs5781-mle/mod3-eliu-
dev/minitorch/fast_ops.py (162)
================================================================================


Parallel loop listing for  Function tensor_map.<locals>._map, /Users/ericliu/Desktop/Github/cornell/cs5781-mle/mod3-eliu-dev/minitorch/fast_ops.py (162)
-----------------------------------------------------------------------------|loop #ID
    def _map(                                                                |
        out: Storage,                                                        |
        out_shape: Shape,                                                    |
        out_strides: Strides,                                                |
        in_storage: Storage,                                                 |
        in_shape: Shape,                                                     |
        in_strides: Strides,                                                 |
    ) -> None:                                                               |
        # TODO: Implement for Task 3.1.e                                     |
        if np.array_equal(out_strides, in_strides) and np.array_equal(       |
            out_shape, in_shape                                              |
        ):                                                                   |
            for i in prange(len(out)):---------------------------------------| #0
                out[i] = fn(in_storage[i])                                   |
        else:                                                                |
            for i in prange(len(out)):---------------------------------------| #1
                out_index: Index = np.empty(MAX_DIMS, np.int32)              |
                in_index: Index = np.empty(MAX_DIMS, np.int32)               |
                to_index(i, out_shape, out_index)                            |
                broadcast_index(out_index, out_shape, in_shape, in_index)    |
                in_position = index_to_position(in_index, in_strides)        |
                out_position = index_to_position(out_index, out_strides)     |
                out[out_position] = fn(in_storage[in_position])              |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 2 parallel for-
loop(s) (originating from loops labelled: #0, #1).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at
/Users/ericliu/Desktop/Github/cornell/cs5781-mle/mod3-eliu-
dev/minitorch/fast_ops.py (178) is hoisted out of the parallel loop labelled #1
(it will be performed before the loop is executed and reused inside the loop):
   Allocation:: out_index: Index = np.empty(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at
/Users/ericliu/Desktop/Github/cornell/cs5781-mle/mod3-eliu-
dev/minitorch/fast_ops.py (179) is hoisted out of the parallel loop labelled #1
(it will be performed before the loop is executed and reused inside the loop):
   Allocation:: in_index: Index = np.empty(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
None
ZIP

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_zip.<locals>._zip,
/Users/ericliu/Desktop/Github/cornell/cs5781-mle/mod3-eliu-
dev/minitorch/fast_ops.py (212)
================================================================================


Parallel loop listing for  Function tensor_zip.<locals>._zip, /Users/ericliu/Desktop/Github/cornell/cs5781-mle/mod3-eliu-dev/minitorch/fast_ops.py (212)
--------------------------------------------------------------------------------|loop #ID
    def _zip(                                                                   |
        out: Storage,                                                           |
        out_shape: Shape,                                                       |
        out_strides: Strides,                                                   |
        a_storage: Storage,                                                     |
        a_shape: Shape,                                                         |
        a_strides: Strides,                                                     |
        b_storage: Storage,                                                     |
        b_shape: Shape,                                                         |
        b_strides: Strides,                                                     |
    ) -> None:                                                                  |
        # TODO: Implement for Task 3.1.                                         |
        if (                                                                    |
            np.array_equal(out_strides, a_strides)                              |
            and np.array_equal(out_strides, b_strides)                          |
            and np.array_equal(out_shape, a_shape)                              |
            and np.array_equal(out_shape, b_shape)                              |
        ):                                                                      |
            for i in prange(len(out)):------------------------------------------| #2
                out[i] = fn(a_storage[i], b_storage[i])                         |
        else:                                                                   |
            for i in prange(len(out)):------------------------------------------| #3
                out_index: Index = np.empty(MAX_DIMS, np.int32)                 |
                a_index: Index = np.empty(MAX_DIMS, np.int32)                   |
                b_index: Index = np.empty(MAX_DIMS, np.int32)                   |
                to_index(i, out_shape, out_index)                               |
                out_position = index_to_position(out_index, out_strides)        |
                broadcast_index(out_index, out_shape, a_shape, a_index)         |
                input_a_position = index_to_position(a_index, a_strides)        |
                broadcast_index(out_index, out_shape, b_shape, b_index)         |
                input_b_position = index_to_position(b_index, b_strides)        |
                out[out_position] = fn(                                         |
                    a_storage[input_a_position], b_storage[input_b_position]    |
                )                                                               |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 2 parallel for-
loop(s) (originating from loops labelled: #2, #3).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at
/Users/ericliu/Desktop/Github/cornell/cs5781-mle/mod3-eliu-
dev/minitorch/fast_ops.py (234) is hoisted out of the parallel loop labelled #3
(it will be performed before the loop is executed and reused inside the loop):
   Allocation:: out_index: Index = np.empty(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at
/Users/ericliu/Desktop/Github/cornell/cs5781-mle/mod3-eliu-
dev/minitorch/fast_ops.py (235) is hoisted out of the parallel loop labelled #3
(it will be performed before the loop is executed and reused inside the loop):
   Allocation:: a_index: Index = np.empty(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at
/Users/ericliu/Desktop/Github/cornell/cs5781-mle/mod3-eliu-
dev/minitorch/fast_ops.py (236) is hoisted out of the parallel loop labelled #3
(it will be performed before the loop is executed and reused inside the loop):
   Allocation:: b_index: Index = np.empty(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
None
REDUCE

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_reduce.<locals>._reduce,
/Users/ericliu/Desktop/Github/cornell/cs5781-mle/mod3-eliu-
dev/minitorch/fast_ops.py (271)
================================================================================


Parallel loop listing for  Function tensor_reduce.<locals>._reduce, /Users/ericliu/Desktop/Github/cornell/cs5781-mle/mod3-eliu-dev/minitorch/fast_ops.py (271)
------------------------------------------------------------------------------|loop #ID
    def _reduce(                                                              |
        out: Storage,                                                         |
        out_shape: Shape,                                                     |
        out_strides: Strides,                                                 |
        a_storage: Storage,                                                   |
        a_shape: Shape,                                                       |
        a_strides: Strides,                                                   |
        reduce_dim: int,                                                      |
    ) -> None:                                                                |
        # TODO: Implement for Task 3.1.                                       |
        reduce_size = a_shape[reduce_dim]                                     |
        for i in prange(len(out)):--------------------------------------------| #5
            out_index: Index = np.zeros(MAX_DIMS, np.int32)-------------------| #4
            to_index(i, out_shape, out_index)                                 |
            out_position = index_to_position(out_index, out_strides)          |
            temp = out[out_position]                                          |
            for s in range(reduce_size):                                      |
                out_index[reduce_dim] = s                                     |
                input_a_position = index_to_position(out_index, a_strides)    |
                temp = fn(temp, a_storage[input_a_position])                  |
                                                                              |
            out[out_position] = temp                                          |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 2 parallel for-
loop(s) (originating from loops labelled: #5, #4).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...

+--5 is a parallel loop
   +--4 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--5 (parallel)
   +--4 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--5 (parallel)
   +--4 (serial)



Parallel region 0 (loop #5) had 0 loop(s) fused and 1 loop(s) serialized as part
 of the larger parallel loop (#5).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at
/Users/ericliu/Desktop/Github/cornell/cs5781-mle/mod3-eliu-
dev/minitorch/fast_ops.py (283) is hoisted out of the parallel loop labelled #5
(it will be performed before the loop is executed and reused inside the loop):
   Allocation:: out_index: Index = np.zeros(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
None
MATRIX MULTIPLY

================================================================================
 Parallel Accelerator Optimizing:  Function _tensor_matrix_multiply,
/Users/ericliu/Desktop/Github/cornell/cs5781-mle/mod3-eliu-
dev/minitorch/fast_ops.py (297)
================================================================================


Parallel loop listing for  Function _tensor_matrix_multiply, /Users/ericliu/Desktop/Github/cornell/cs5781-mle/mod3-eliu-dev/minitorch/fast_ops.py (297)
------------------------------------------------------------------------------------------|loop #ID
def _tensor_matrix_multiply(                                                              |
    out: Storage,                                                                         |
    out_shape: Shape,                                                                     |
    out_strides: Strides,                                                                 |
    a_storage: Storage,                                                                   |
    a_shape: Shape,                                                                       |
    a_strides: Strides,                                                                   |
    b_storage: Storage,                                                                   |
    b_shape: Shape,                                                                       |
    b_strides: Strides,                                                                   |
) -> None:                                                                                |
    """NUMBA tensor matrix multiply function.                                             |
                                                                                          |
    Should work for any tensor shapes that broadcast as long as                           |
                                                                                          |
    ```                                                                                   |
    assert a_shape[-1] == b_shape[-2]                                                     |
    ```                                                                                   |
                                                                                          |
    Optimizations:                                                                        |
                                                                                          |
    * Outer loop in parallel                                                              |
    * No index buffers or function calls                                                  |
    * Inner loop should have no global writes, 1 multiply.                                |
                                                                                          |
                                                                                          |
    Args:                                                                                 |
    ----                                                                                  |
        out (Storage): storage for `out` tensor                                           |
        out_shape (Shape): shape for `out` tensor                                         |
        out_strides (Strides): strides for `out` tensor                                   |
        a_storage (Storage): storage for `a` tensor                                       |
        a_shape (Shape): shape for `a` tensor                                             |
        a_strides (Strides): strides for `a` tensor                                       |
        b_storage (Storage): storage for `b` tensor                                       |
        b_shape (Shape): shape for `b` tensor                                             |
        b_strides (Strides): strides for `b` tensor                                       |
                                                                                          |
    Returns:                                                                              |
    -------                                                                               |
        None : Fills in `out`                                                             |
                                                                                          |
    """                                                                                   |
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0                                |
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0                                |
                                                                                          |
    # TODO: Implement for Task 3.2.                                                       |
    for n in prange(out_shape[0]):--------------------------------------------------------| #8
        for i in prange(out_shape[1]):----------------------------------------------------| #7
            for j in prange(out_shape[2]):------------------------------------------------| #6
                temp = 0.0                                                                |
                for k in range(a_shape[-1]):                                              |
                    a_pos = n * a_batch_stride + i * a_strides[1] + k * a_strides[2]      |
                    b_pos = n * b_batch_stride + k * b_strides[1] + j * b_strides[2]      |
                    temp += a_storage[a_pos] * b_storage[b_pos]                           |
                out_pos = n * out_strides[0] + i * out_strides[1] + j * out_strides[2]    |
                out[out_pos] = temp                                                       |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 2 parallel for-
loop(s) (originating from loops labelled: #8, #7).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...

+--8 is a parallel loop
   +--7 --> rewritten as a serial loop
      +--6 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--8 (parallel)
   +--7 (parallel)
      +--6 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--8 (parallel)
   +--7 (serial)
      +--6 (serial)



Parallel region 0 (loop #8) had 0 loop(s) fused and 2 loop(s) serialized as part
 of the larger parallel loop (#8).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None
```

# Task 3.4 Performance

Please see [cuda_ops.py](minitorch/cuda_ops.py) comments for discussion of cuda optimizations.

![performance](performance.png)
*Performance comparison showing execution time between Fast CPU (red) and GPU (blue) implementations for matrix multiplication with increasing matrix sizes*


# Task 3.5 Training Results

## Simple: Fast (CPU)

- Epochs: 500
- Hidden: 100
- Learning Rate: 0.05
- Time/epoch: 0.1243471155166626 seconds per epoch

Epoch|Loss|Correct|Time/epoch
---|---|---|---|
0|5.983959202|25|0s
10|1.689794821|47|0.11034717559814453s
20|1.276932137|49|0.11404163837432861s
30|1.304914993|49|0.11190121173858643s
40|0.9542397208|50|0.11102762222290039s
50|0.9333790125|50|0.11245923042297364s
60|0.600132099|50|0.1200181484222412s
70|0.1474278464|50|0.17576940059661866s
80|0.8160415787|50|0.1520390510559082s
90|1.100209747|50|0.11304168701171875s
100|0.8374354241|50|0.11439838409423828s
110|0.1243886392|50|0.11232337951660157s
120|0.2061082738|50|0.11224842071533203s
130|0.2839728006|50|0.11165683269500733s
140|0.05422736696|50|0.11135692596435547s
150|0.05294018567|50|0.11137504577636718s
160|0.6016158786|50|0.11185805797576905s
170|0.6139783024|50|0.18423881530761718s
180|0.5391801108|50|0.16409525871276856s
190|0.4379799555|50|0.11205580234527587s
200|0.07994356459|50|0.11220402717590332s
210|0.1797245341|50|0.1134479284286499s
220|0.108639648|50|0.11297378540039063s
230|0.0166599826|50|0.11589128971099853s
240|0.00829895679|50|0.11174936294555664s
250|0.3976635167|50|0.11252362728118896s
260|0.6217575435|50|0.11061007976531982s
270|0.08115246669|50|0.17135224342346192s
280|0.519001364|50|0.1776569128036499s
290|0.3424243444|50|0.11214635372161866s
300|0.6683690714|50|0.11263597011566162s
310|0.004814308025|50|0.11160850524902344s
320|0.5240107461|50|0.11238043308258057s
330|0.5001120828|50|0.11206614971160889s
340|0.0002156912024|50|0.11228115558624267s
350|0.6429862404|50|0.11234908103942871s
360|0.01315268802|50|0.11141483783721924s
370|0.228879589|50|0.1551410436630249s
380|0.6024924373|50|0.18997230529785156s
390|0.7508733573|50|0.11256163120269776s
400|0.9104007912|49|0.1117234468460083s
410|0.3809080286|50|0.11161751747131347s
420|0.1726508084|50|0.1118014097213745s
430|0.2903317269|50|0.11047680377960205s
440|0.0007105757024|50|0.11589360237121582s
450|0.08152785283|50|0.11297821998596191s
460|0.001926611436|50|0.11177456378936768s
470|0.5639963416|50|0.12604067325592042s
480|0.09627998962|50|0.22108333110809325s
490|0.2701012651|50|0.11236453056335449s
500|0.2991206766|50|0.11238286495208741s

## Simple: GPU
- Epochs: 500
- Hidden: 100
- Learning Rate: 0.05
- Time/epoch: 1.642142177581787 seconds per epoch

Epoch|Loss|Correct|Time/epoch
---|---|---|---|
0|3.399480392|42|0s
10|1.484362733|49|1.6048332691192626s
20|0.7692891493|49|1.5905674934387206s
30|1.281336531|50|1.6851417541503906s
40|0.5249616147|49|1.6098650693893433s
50|1.043396476|50|1.605553102493286s
60|1.258632155|50|1.6924528121948241s
70|0.01730257359|50|1.6161447763442993s
80|0.5947965265|50|1.6369374513626098s
90|0.1820262638|50|1.7923821210861206s
100|0.2093103165|50|1.6096657514572144s
110|0.3999902216|50|1.6170504331588744s
120|0.5500186974|50|1.7288958787918092s
130|0.1476296459|50|1.6308682918548585s
140|0.06465967959|50|1.6291401147842408s
150|0.09633791353|50|1.7147844076156615s
160|0.01172721006|50|1.6112353563308717s
170|0.3321925174|50|1.606659722328186s
180|0.2864550647|50|1.6433016061782837s
190|0.09154390186|50|1.6401839971542358s
200|0.2048631752|50|1.592628502845764s
210|0.1193195305|50|1.6183753728866577s
220|0.2563071734|50|1.6574657201766967s
230|0.3926502951|50|1.6085356235504151s
240|0.1334355959|50|1.603399658203125s
250|0.3635272123|50|1.7764771699905395s
260|0.04429620688|50|1.5997466087341308s
270|0.1900938354|50|1.6068440914154052s
280|0.1021808673|50|1.696448850631714s
290|0.006340794589|50|1.6041100025177002s
300|0.09853392298|50|1.6156697750091553s
310|0.07854211968|50|1.6904087305068969s
320|0.02469522201|50|1.6188988208770752s
330|0.129582614|50|1.619704008102417s
340|0.03397452119|50|1.6918120861053467s
350|0.01409380039|50|1.604250168800354s
360|0.01157880434|50|1.6139886617660522s
370|0.02942712104|50|1.6925230026245117s
380|0.02494831614|50|1.6125811338424683s
390|0.06848007009|50|1.5956628561019897s
400|0.08364738374|50|1.6791917562484742s
410|0.09449190929|50|1.7167429208755494s
420|0.06064694457|50|1.6013405561447143s
430|0.1007206504|50|1.6210046052932738s
440|0.03215116118|50|1.669840121269226s
450|0.02203520026|50|1.609895157814026s
460|0.007971371492|50|1.611148428916931s
470|0.0001893062622|50|1.7030783653259278s
480|0.01921817295|50|1.605277132987976s
490|0.07876474742|50|1.607790207862854s
500|0.1243594674|50|1.696605372428894s

## Split: Fast (CPU)
- Epochs: 500
- Hidden: 100
- Learning Rate: 0.05
- Time/epoch: 0.12500274991989135 seconds per epoch

Epoch|Loss|Correct|Time/epoch
---|---|---|---
0|6.730808772|33|0s
10|5.381105152|40|0.11638319492340088s
20|4.697563946|40|0.11377236843109131s
30|3.740515775|41|0.11406054496765136s
40|3.409539562|46|0.11244828701019287s
50|1.144918334|40|0.12025308609008789s
60|4.174225504|47|0.11237850189208984s
70|2.192036028|48|0.11178891658782959s
80|1.571420573|48|0.21355869770050048s
90|1.859916262|49|0.13461191654205323s
100|2.281571092|49|0.1164445161819458s
110|1.328391961|49|0.11289145946502685s
120|1.538352315|50|0.11276731491088868s
130|2.595673757|50|0.11279850006103516s
140|0.2065882024|50|0.11177327632904052s
150|1.435092065|49|0.11212220191955566s
160|0.7824790258|50|0.11195688247680664s
170|1.048179241|50|0.11250336170196533s
180|0.6944126033|49|0.19604794979095458s
190|0.6699664152|50|0.1504455327987671s
200|0.5166320679|50|0.11206164360046386s
210|0.8782551751|49|0.11099865436553955s
220|0.9864642164|50|0.11053700447082519s
230|0.7902335582|50|0.11087601184844971s
240|0.7634741032|49|0.11093010902404785s
250|0.6313371159|50|0.11346278190612794s
260|0.2336113774|50|0.11216745376586915s
270|0.6472391199|50|0.10973968505859374s
280|0.6654045879|50|0.1999392032623291s
290|0.6752085079|49|0.1540457487106323s
300|0.5017316544|49|0.11281402111053467s
310|0.004114960514|50|0.1120058536529541s
320|0.08585808699|49|0.11280186176300049s
330|0.3589201317|49|0.11247818470001221s
340|1.556660438|49|0.1123878002166748s
350|0.1057537465|50|0.11325891017913818s
360|0.2549828912|50|0.11199657917022705s
370|1.024008686|49|0.11215887069702149s
380|0.7287817601|50|0.19395880699157714s
390|0.3230464326|50|0.15862584114074707s
400|0.3783589949|50|0.11187663078308105s
410|0.3945517543|50|0.11239078044891357s
420|0.1784443471|50|0.11227502822875976s
430|0.1315917716|50|0.11083266735076905s
440|0.1704087783|50|0.11190407276153565s
450|0.556088425|50|0.11183950901031495s
460|0.1837145418|50|0.11123573780059814s
470|0.09935413611|50|0.11256439685821533s
480|0.3820630246|50|0.1657099485397339s
490|0.5813537854|50|0.18003394603729247s
500|0.09569022448|50|0.11322324275970459s

## Split: GPU
- Epochs: 500
- Hidden: 100
- Learning Rate: 0.05
- Time/epoch: 1.6348775901794435 seconds per epoch

Epoch|Loss|Correct|Time/epoch
---|---|---|---
0|6.11833288|36|0s
10|4.038004681|39|1.6362900495529176s
20|4.878591895|40|1.7540507793426514s
30|3.991856921|41|1.5953732490539552s
40|2.987850945|46|1.6463940382003783s
50|2.240311631|44|1.6288623809814453s
60|1.813302888|48|1.590417456626892s
70|1.047957461|48|1.5851238012313842s
80|1.645138818|49|1.679257082939148s
90|2.045077222|50|1.5917662382125854s
100|2.627573345|50|1.5984307289123536s
110|0.9071461389|49|1.6737948417663575s
120|0.8104841652|49|1.5930098295211792s
130|0.4003687679|50|1.5872062206268311s
140|0.7658324939|50|1.6695320367813111s
150|0.910363142|50|1.5902894735336304s
160|0.357760068|50|1.605583667755127s
170|0.7047600172|50|1.64179847240448s
180|0.427960941|50|1.7376643896102906s
190|0.3259309731|50|1.6085569143295289s
200|0.1513782593|50|1.6080567121505738s
210|0.4455421888|50|1.6903769254684449s
220|0.2825594905|50|1.6108615159988404s
230|0.5552865791|50|1.6225314617156983s
240|0.1758135255|50|1.7006522178649903s
250|0.4659165503|50|1.5997370481491089s
260|0.1268871055|50|1.6129345417022705s
270|0.3527209857|50|1.6829018354415894s
280|0.1893544501|50|1.6127679824829102s
290|0.03468962827|50|1.6189451694488526s
300|0.1725514793|50|1.6834758996963501s
310|0.05017697251|50|1.595018243789673s
320|0.2494315821|50|1.5948840379714966s
330|0.5675629977|50|1.6259076118469238s
340|0.0400914962|50|1.6557308912277222s
350|0.1537384783|50|1.6796670436859131s
360|0.17807979|50|1.6317053318023682s
370|0.1272283296|50|1.6748557329177856s
380|0.3424594176|50|1.6103210926055909s
390|0.265729021|50|1.6158881902694702s
400|0.2797142011|50|1.6835984706878662s
410|0.246251931|50|1.606606698036194s
420|0.3179972246|50|1.6054295539855956s
430|0.03474858226|50|1.6935835838317872s
440|0.2606152726|50|1.6022410631179809s
450|0.1889184361|50|1.602644968032837s
460|0.1182577323|50|1.6812785148620606s
470|0.05508731628|50|1.6141746759414672s
480|0.09509317858|50|1.6203595399856567s
490|0.08518690505|50|1.6283952713012695s
500|0.08438569397|50|1.6649460315704345s

## XOR: Fast (CPU)
- Epochs: 500
- Hidden: 100
- Learning Rate: 0.05
- Time/epoch: 0.1250982985496521 seconds per epoch

Epoch|Loss|Correct|Time/epoch
---|---|---|---
0|7.110668242|36|0s
10|2.697171961|38|0.11197860240936279s
20|6.856946146|41|0.1115910530090332s
30|3.379054982|42|0.11433947086334229s
40|2.961727763|44|0.12241470813751221s
50|4.64932841|43|0.1131822109222412s
60|4.134135851|45|0.11351630687713624s
70|3.894672077|44|0.11339442729949951s
80|2.555212005|44|0.16170814037322997s
90|3.223280039|45|0.1864617109298706s
100|3.713546488|45|0.10949153900146484s
110|3.052351645|45|0.11127598285675049s
120|1.781453987|46|0.11084635257720947s
130|2.351371654|46|0.11015162467956544s
140|2.167693917|44|0.11284878253936767s
150|3.927237634|45|0.11547191143035888s
160|1.277283985|47|0.11129572391510009s
170|2.948237211|45|0.11168508529663086s
180|2.426763182|47|0.13962311744689943s
190|3.135904011|46|0.2093740463256836s
200|0.6139006997|46|0.11192893981933594s
210|2.00311498|48|0.11172313690185547s
220|3.166250683|46|0.11356890201568604s
230|2.318865236|47|0.11166071891784668s
240|1.105821267|48|0.11048483848571777s
250|0.3711050015|48|0.11161966323852539s
260|1.693831207|48|0.11319797039031983s
270|2.525134634|49|0.11215107440948487s
280|1.239087751|48|0.13112046718597412s
290|0.2837965504|49|0.22552402019500734s
300|1.359708362|47|0.11349303722381592s
310|0.4234008045|49|0.1120689868927002s
320|0.747885976|48|0.11093945503234863s
330|1.695356407|49|0.11508800983428955s
340|2.20082299|48|0.11119530200958253s
350|1.177997905|47|0.11286287307739258s
360|0.5794589971|50|0.11040191650390625s
370|3.366571867|48|0.11253924369812011s
380|1.378320751|49|0.12016911506652832s
390|0.6702821397|50|0.21921250820159913s
400|1.01963241|48|0.1291297435760498s
410|0.4644791136|49|0.11239795684814453s
420|1.014935615|48|0.11062209606170655s
430|1.667186145|49|0.11270129680633545s
440|0.5866404788|50|0.11289830207824707s
450|1.728085078|49|0.11356985569000244s
460|0.5469247792|49|0.11210792064666748s
470|1.50784356|49|0.11399660110473633s
480|0.1726408235|49|0.1157649040222168s
490|2.503419199|49|0.19405584335327147s
500|0.1120435245|49|0.14606943130493164s

## XOR: GPU
- Epochs: 500
- Hidden: 100
- Learning Rate: 0.05
- Time/epoch: 1.640591947555542 seconds per epoch


Epoch|Loss|Correct|Time/epoch
---|---|---|---
0|6.636588539|34|0s
10|5.649019563|41|1.6417090892791748s
20|4.075416355|47|1.6743024110794067s
30|3.542651773|48|1.6071485996246337s
40|3.084698037|47|1.6188102960586548s
50|2.336894017|47|1.6949307680130006s
60|3.979111425|48|1.6048071384429932s
70|1.555346471|47|1.5979215621948242s
80|1.212671708|48|1.6867366313934327s
90|3.825101333|48|1.6124886751174927s
100|1.45623956|47|1.5952327966690063s
110|1.190392153|48|1.7076795816421508s
120|1.074154927|48|1.708490538597107s
130|2.381638379|47|1.6032592535018921s
140|1.571289119|48|1.6894803047180176s
150|0.7875681453|49|1.607517123222351s
160|1.121112484|48|1.5978452205657958s
170|0.5415620927|48|1.6602060317993164s
180|1.437631352|49|1.6362502336502076s
190|1.453685095|49|1.6108354568481444s
200|0.9680236733|50|1.6088305711746216s
210|1.595562947|50|1.6946206569671631s
220|1.589358556|49|1.6006510496139525s
230|0.464386311|48|1.6090584516525268s
240|1.063908446|50|1.6895127773284913s
250|1.425212619|50|1.616690993309021s
260|0.8899222547|50|1.6060304403305055s
270|0.7837272019|50|1.6995465278625488s
280|0.6587723354|50|1.7201547622680664s
290|1.222991761|50|1.6103561162948608s
300|0.7153203174|50|1.6686444997787475s
310|0.1641153726|50|1.6465285301208497s
320|0.6640792693|50|1.6090454816818238s
330|0.8552519421|50|1.628679585456848s
340|0.8685728697|50|1.6819869518280028s
350|1.073407546|50|1.6053407669067383s
360|0.882126828|50|1.6093369483947755s
370|0.6200451858|50|1.6930992603302002s
380|0.5430700716|50|1.6049522399902343s
390|0.2043471028|50|1.6019614934921265s
400|0.4770269618|50|1.6991549253463745s
410|0.5904314889|50|1.61409649848938s
420|0.3134007743|50|1.6095786094665527s
430|0.2954438839|50|1.6946695804595948s
440|0.3465088769|50|1.6031892061233521s
450|0.1472755809|50|1.6877193689346313s
460|0.5159378243|50|1.6875910997390746s
470|0.1863390918|50|1.6043274402618408s
480|0.7195316875|50|1.6020952224731446s
490|0.2118257294|50|1.6330425977706908s
500|0.02715931878|50|1.6334530115127563s

## XOR Large Model: Fast (CPU)
- Epochs: 500
- Hidden: 300
- Learning Rate: 0.05
- Time/epoch: 0.5408324394226074 seconds per epoch

### Fast (CPU)

Epoch|Loss|Correct|Time/epoch
---|---|---|---
0|28.98668523|29|0s
10|4.299792058|47|0.570708441734314s
20|3.811682787|45|0.4922958850860596s
30|0.3044211631|44|0.6070119380950928s
40|0.9370687966|48|0.49802298545837403s
50|1.962794628|49|0.6075644016265869s
60|2.200975942|45|0.49272520542144777s
70|0.3579597124|50|0.5631260395050048s
80|1.392908592|50|0.6532637119293213s
90|0.9168985182|50|0.5287804126739502s
100|1.120632979|50|0.5721905946731567s
110|0.332435319|50|0.4903621435165405s
120|1.46008839|48|0.5988436460494995s
130|0.5086771984|50|0.4892911195755005s
140|0.228212362|48|0.5997619152069091s
150|0.7300736345|49|0.4946368932723999s
160|1.120247341|50|0.5261472702026367s
170|0.6878798886|50|0.5670496702194214s
180|0.7516644551|50|0.4864538908004761s
190|0.5897357591|50|0.6040047168731689s
200|0.2367533791|50|0.4842890024185181s
210|0.07592948786|48|0.6088897228240967s
220|0.08316808108|50|0.48607895374298093s
230|0.8456091272|50|0.5379128217697143s
240|0.7442752941|50|0.5498218536376953s
250|0.57991723|50|0.4862103223800659s
260|0.2004231391|50|0.6012647390365601s
270|0.6095751598|50|0.48376443386077883s
280|0.1266897116|50|0.5982664346694946s
290|0.2357829436|50|0.4839579343795776s
300|0.1361858314|50|0.5239962577819824s
310|0.4462690272|50|0.5604985237121582s
320|0.04315451334|50|0.48417255878448484s
330|0.2445167649|50|0.6030301332473755s
340|0.3213734273|50|0.48463280200958253s
350|0.6127769219|50|0.6039200305938721s
360|0.2608957796|50|0.4903119564056396s
370|0.3961795036|50|0.5165516853332519s
380|0.2078295921|50|0.5756758451461792s
390|0.07982494337|50|0.4868889093399048s
400|0.2174853679|50|0.5981239080429077s
410|0.08407274113|50|0.4855510711669922s
420|0.1383777283|50|0.5992023229599s
430|0.1222534238|50|0.4918036460876465s
440|0.2684398538|50|0.49992666244506834s
450|0.2112125064|50|0.5906825065612793s
460|0.4161603617|50|0.4846641540527344s
470|0.1284016249|50|0.6071902990341187s
480|0.2975918924|50|0.4893318176269531s
490|0.1756946109|50|0.6092713356018067s
500|0.4196845763|50|0.49349844455718994s

## XOR Large Model: GPU
- Epochs: 500
- Hidden: 300
- Learning Rate: 0.05
- Seconds per epoch: 1.8861037230491637 seconds per epoch

Epoch|Loss|Correct|Time/epoch
---|---|---|---
0|16.97916534|27|0s
10|7.721242584|35|1.8897896528244018s
20|2.483018157|48|1.960396409034729s
30|3.34381821|46|1.9154447078704835s
40|0.7742425643|47|1.8720311880111695s
50|2.114281794|47|1.881920576095581s
60|0.734652527|48|1.8592427492141723s
70|1.795046191|47|1.8611066102981568s
80|0.9103303188|48|1.8674829483032227s
90|0.9126002112|49|1.8691225051879883s
100|1.330158758|50|1.8516199827194213s
110|0.8141874366|49|1.8627242565155029s
120|0.8176360859|50|1.8741047620773315s
130|0.2557790008|49|1.8641999006271361s
140|1.321572261|48|1.877837610244751s
150|0.2109924833|48|1.8603381156921386s
160|0.3853922697|49|1.8950580596923827s
170|0.05714690471|49|1.950979733467102s
180|1.209785734|49|1.9054524421691894s
190|0.654277474|49|1.8500133752822876s
200|1.128487266|49|1.9271531343460082s
210|0.8744482312|48|1.8448725700378419s
220|0.6178706088|49|1.923042321205139s
230|0.07923439302|49|1.8387794017791748s
240|0.8829787997|50|1.9308315515518188s
250|1.349258592|50|1.8409925937652587s
260|0.5239303536|50|1.911802339553833s
270|0.6235712166|49|1.864223575592041s
280|0.6419537819|50|1.9472135543823241s
290|0.2445872811|50|1.8457875490188598s
300|0.06545038131|50|1.932627010345459s
310|0.520972828|49|1.938916230201721s
320|0.1387220498|50|1.9285512685775756s
330|0.5248652391|49|1.8368083953857421s
340|0.1160881342|50|1.9225331783294677s
350|0.740943231|49|1.84727201461792s
360|0.02288965016|49|1.9164968252182006s
370|0.5149230587|49|1.8270277023315429s
380|0.2076596606|50|1.9087090492248535s
390|0.5745539362|50|1.8214847326278687s
400|0.7368350959|50|1.9215890884399414s
410|0.8260409887|50|1.8398155689239502s
420|0.1202753903|50|1.9071176290512084s
430|0.3035317956|50|1.8448238611221313s
440|0.4083462863|50|1.8984958171844482s
450|0.6698371684|50|1.852009654045105s
460|0.02447308041|50|1.991407608985901s
470|0.9592122454|50|1.913141131401062s
480|0.0205813429|50|1.8539097785949707s
490|0.6792412194|50|1.9137442111968994s
500|0.01438174224|50|1.8451412200927735s
