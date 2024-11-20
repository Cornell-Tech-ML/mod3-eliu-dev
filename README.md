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

## Simple

### Fast (CPU)

- Hidden: 100
- Learning Rate: 0.05

Epoch|Loss|Correct
---|---|---
0|5.183147992|41
10|2.006881006|49
20|2.069937409|49
30|0.5863635469|49
40|1.452448572|49
50|0.3110547472|49
60|0.4185370148|49
70|0.6771303265|49
80|0.5118949833|49
90|0.2767845558|50
100|0.7704364863|50
110|0.06864466411|50
120|0.9866397891|50
130|0.2727201438|50
140|0.1353256067|50
150|0.03040099753|50
160|1.045750144|49
170|0.02788914214|50
180|0.1178521504|50
190|0.07043705647|50
200|0.629406955|50
210|0.1057739743|50
220|0.08908092271|50
230|0.03230743921|50
240|1.002573325|50
250|0.50334627|50
260|0.01217520156|50
270|0.04157440544|50
280|0.007655597035|50
290|0.4375375907|50
300|0.01417541761|50
310|0.001253660116|50
320|0.01574114688|50
330|0.01248749147|50
340|0.4455836424|50
350|0.1474577889|50
360|0.4555137851|50
370|0.3884117999|50
380|0.5202511626|50
390|0.3584482185|50
400|0.0918554724|50
410|0.04560296694|50
420|0.1980824042|50
430|0.003144781807|50
440|0.01154478271|50
450|0.3886403031|50
460|0.000818495419|50
470|0.1779788747|50
480|0.298070195|50
490|0.3229422032|50

### GPU

- Hidden: 100
- Learning Rate: 0.01

Epoch|Loss|Correct
---|---|---
0|5.809264861|39
10|2.048606279|48
20|1.519921271|49
30|1.675174859|50
40|1.643199942|50
50|1.052151287|50
60|1.481459855|50
70|1.173704446|50
80|1.170627244|50
90|0.8470480841|50
100|0.6882103293|50
110|0.8644310915|50
120|1.285718963|50
130|0.9843193718|50
140|0.7646657301|50
150|0.6698233339|50
160|1.083846367|50
170|0.8250878406|50
180|0.09774261604|50
190|0.3765100331|50
200|0.8183113561|50
210|0.07129557384|50
220|0.431866608|50
230|0.6961247589|50
240|0.1739085951|50
250|0.07931649621|50
260|0.3326733248|50
270|0.7335481882|50
280|0.8100440645|50
290|0.2414272814|50
300|0.7658576178|50
310|0.3480517281|50
320|0.5006124042|50
330|0.3071422914|50
340|0.1769691282|50
350|0.04731690217|50
360|0.5538661735|50
370|0.5767271169|50
380|0.04709981398|50
390|0.9738980188|50
400|0.6501300661|50
410|0.05533086926|50
420|0.3614379459|50
430|0.1691490183|50
440|0.5080871144|50
450|0.7079013935|50
460|0.230274051|50
470|0.6329263035|50
480|0.2646207037|50
490|0.3777088517|50

## Split

### Fast (CPU)
- Hidden: 100
- Learning Rate: 0.05

Epoch|Loss|Correct
---|---|---
0|6.85956399|30
10|4.977850826|41
20|4.660836699|37
30|4.893449374|43
40|4.588566103|48
50|3.338016835|49
60|2.921386467|47
70|2.845455446|49
80|2.361650695|47
90|2.513458846|44
100|1.289247675|50
110|2.34706356|48
120|0.7966038635|49
130|2.095222211|48
140|2.142269325|44
150|1.923991438|50
160|2.275182016|49
170|1.211060485|50
180|2.872514041|50
190|0.7920737853|50
200|1.383167381|50
210|2.126805587|50
220|2.995781332|47
230|1.500281381|50
240|1.836518686|50
250|1.62917353|48
260|1.549083693|50
270|1.024334395|50
280|0.5788142348|50
290|2.336578484|50
300|1.357966617|50
310|0.8389680395|50
320|1.400883696|50
330|0.4714169275|50
340|0.3487766546|49
350|1.177574888|49
360|0.6823810714|50
370|1.805249559|50
380|0.17747606|49
390|0.7266019848|50
400|0.6140351929|50
410|0.5551846677|50
420|0.5941607145|50
430|0.9065531846|50
440|0.8224411073|50
450|0.3189156756|49
460|0.1887757494|49
470|0.140387459|50
480|1.840276808|50
490|0.5817390929|50

### GPU
- Hidden: 100
- Learning Rate: 0.05

Epoch|Loss|Correct
---|---|---
0|5.683762087|32
10|3.487924676|43
20|5.373540871|42
30|4.677735676|44
40|3.212549686|46
50|2.706751889|46
60|2.42845748|48
70|2.390780224|49
80|1.998732439|47
90|2.213281715|49
100|1.590470459|50
110|1.721910549|49
120|0.5993762434|49
130|0.6517542508|50
140|1.288626735|50
150|1.429527638|49
160|0.5117240247|48
170|1.613304023|48
180|1.245613826|49
190|1.115668655|50
200|0.3387142119|50
210|0.6880020961|50
220|2.17399776|47
230|0.6255060921|50
240|0.4776187995|50
250|1.213305988|50
260|0.8546643635|50
270|1.427916971|49
280|0.3659060001|49
290|0.5815064868|50
300|0.3243066116|50
310|0.5416671855|50
320|1.645580848|48
330|0.3567352612|50
340|1.083129702|50
350|0.2109481137|50
360|0.4172372164|50
370|0.4443090069|50
380|0.2071719479|50
390|0.2294312236|50
400|0.5245797534|50
410|0.04097977059|50
420|0.07390725713|50
430|0.09729824908|50
440|0.5440881918|50
450|0.1949956105|50
460|0.2820969219|50
470|0.6269811281|50
480|0.4545980416|50
490|0.3445936414|50

## XOR

### Fast (CPU)

- Hidden: 100
- Learning Rate: 0.05

Epoch|Loss|Correct
---|---|---
0|6.507335105|32
10|3.737375441|44
20|1.915978687|48
30|3.932262825|46
40|2.565735838|50
50|1.644499615|48
60|2.353152406|47
70|1.423468857|49
80|2.47834959|48
90|1.012508531|50
100|1.659269687|49
110|0.8634472289|49
120|2.037713365|50
130|0.7259646634|49
140|0.8856164961|49
150|0.955973419|49
160|0.9973599615|50
170|1.530619865|50
180|0.426614713|50
190|0.1735001221|50
200|0.8447484202|50
210|0.4409387842|50
220|0.2934022721|50
230|0.7518833954|50
240|0.700402582|50
250|0.8593544181|50
260|0.3928658568|50
270|0.1772091011|50
280|0.4621756901|50
290|0.1521182396|50
300|0.4031963617|50
310|0.6897646743|50
320|0.1400317955|50
330|0.2558331354|50
340|0.1749482446|50
350|0.3298142049|50
360|0.1220415408|50
370|0.04258274028|50
380|0.1893042415|50
390|0.4068000962|50
400|0.4741540028|50
410|0.2044835019|50
420|0.1046334338|50
430|0.0948726311|50
440|0.07072209719|50
450|0.1736008839|50
460|0.2813472819|50
470|0.113802094|50
480|0.1284202241|50
490|0.2156106381|50

### GPU

Hidden: 100
Learning Rate: 0.05

Epoch|Loss|Correct
---|---|---
0|7.413267881|23
10|4.225882467|43
20|2.672725719|45
30|2.884055881|47
40|4.572642827|42
50|4.633493097|46
60|1.629983886|45
70|2.52660384|45
80|2.082247036|48
90|2.223015581|47
100|1.785409569|48
110|0.8142699756|48
120|0.6873832164|48
130|1.529018701|45
140|2.305501096|49
150|1.61620122|49
160|1.586659968|49
170|1.573920475|48
180|0.5738460902|49
190|0.2873111967|50
200|0.6292005509|48
210|1.362497412|49
220|2.255620562|49
230|1.015816584|49
240|1.025305239|50
250|1.32981842|49
260|1.939630803|49
270|0.4439616595|49
280|0.3281999197|50
290|0.2504660719|49
300|1.020054111|49
310|0.8167229005|49
320|0.9610491689|49
330|0.1382765349|49
340|1.478419079|49
350|0.8184105152|49
360|1.350380874|49
370|0.3982806739|49
380|0.6623894998|49
390|0.4831676146|50
400|0.1922058959|49
410|0.7754252698|49
420|0.264851796|49
430|0.2977891356|50
440|1.278263641|49
450|0.5010034164|50
460|0.1364420106|49
470|0.007205155934|49
480|0.2504399893|49
490|0.3981666473|49

## Large Model (XOR)

- Hidden: 200
- Learning Rate: 0.05
- Seconds per epoch: **0.3397 seconds/epoch**

### Fast (CPU)

Epoch|Loss|Correct
---|---|---
0|8.939340554|31
10|6.712564517|39
20|1.324105327|46
30|3.07147117|46
40|2.041332193|48
50|0.444856897|48
60|2.016095736|49
70|1.435443952|49
80|0.6484578044|50
90|0.2794413048|48
100|0.5592202067|49
110|0.2565677076|49
120|1.469310722|49
130|0.518286502|49
140|1.037931082|49
150|0.2018874967|50
160|1.11724378|49
170|0.5718100829|49
180|1.406851986|49
190|0.36600887|49
200|1.317066569|49
210|1.095256624|49
220|0.1798959716|50
230|0.3383245933|49
240|1.297873603|49
250|1.017628792|49
260|1.591569531|49
270|0.8283397847|50
280|1.604172632|49
290|0.04733192136|49
300|0.1308492065|50
310|0.525749327|50
320|0.2091831424|49
330|0.0820400483|50
340|1.119216253|49
350|0.1582050228|49
360|0.1816903909|50
370|1.394580252|49
380|0.2605046321|49
390|0.112645666|49
400|0.4242849527|50
410|0.6229721636|49
420|0.09518653473|49
430|0.161698001|50
440|0.1794845063|49
450|1.118531687|49
460|0.206198671|49
470|0.6478649727|49
480|1.585431598|49
490|1.101993675|49

### GPU

- Hidden: 200
- Learning Rate: 0.05
- Seconds per epoch: **1.7413 seconds/epoch**

Epoch|Loss|Correct
---|---|---
0|10.4560515|32
10|7.920495607|42
20|12.72644841|41
30|2.382790234|47
40|2.603437932|49
50|2.367972231|46
60|2.154782412|48
70|2.574751623|47
80|2.501650454|47
90|1.40920287|49
100|1.447996133|49
110|1.081265005|49
120|1.181311479|50
130|0.5913031494|49
140|0.7888440511|50
150|0.205659468|49
160|0.1342774263|50
170|1.819672028|48
180|0.5276785071|50
190|0.9505865709|50
200|0.5966108973|50
210|0.5179291503|50
220|0.2352510895|50
230|0.8064998418|50
240|0.4766026867|50
250|0.7329017994|50
260|0.3271403403|50
270|0.2316107746|50
280|0.2710407606|50
290|0.1884111786|50
300|0.2174918515|50
310|0.2873740797|50
320|0.4926562269|50
330|0.2227337367|50
340|0.1695391914|50
350|0.1272271156|50
360|0.2711674715|50
370|0.2581194227|50
380|0.4282051259|50
390|0.3027662443|50
400|0.3239152775|50
410|0.1779049608|50
420|0.1303425484|50
430|0.1584161429|50
440|0.06498841357|50
450|0.302165472|50
460|0.3172927104|50
470|0.1266272448|50
480|0.1520009645|50
490|0.1285715324|50