
# TP01

In all the code sections above, the variable `pol` is the cache strategy to use. LRU=0, LFU=1

## Step 1
### Code
```c
    for (i = 0; i < TM; ++ i) {
        for (j = 0; j < TM; ++ j) {
            for (k = 0; k < TM; ++ k) {


                // A[i][k] * B[k][j]
                adra_el = ADRA + i * TM * TELEM + k * TELEM;
                adrb_el = ADRB + k * TM * TELEM + j * TELEM;

                // cache access
                if (pol == 0) {
                    NhitA += cache (adra_el, init_lru, maj_lru);
                    NhitB += cache (adrb_el, init_lru, maj_lru);
                } else {
                    NhitB += cache (adra_el, init_lfu, maj_lfu);
                    NhitA += cache (adrb_el, init_lfu, maj_lfu);
                }
            }
        }
    }
```

### Results with LRU cache strategy
```
valeurs mesurés LRU	
succès A	succès B
99.902344	99.901962
99.084427	0.000%
99.968750	93.745018
99.545280	0.000%
99.979172	93.747780
99.984375	93.748749
96.102745	57.618046
99.775696	0.000%
93.690163	2.918454
93.607346	0.000%
96.734291	0.000%
93.639015	0.000%
```

## Step 2
```c
    for (i = 0; i < TM; ++ i) {
        for (j = 0; j < TM; j += 16) {
            for (k = 0; k < TM; ++ k) {
                int m = j + 16;
                for (jk = j; jk < m; ++ jk) {
                    // A[i][k] * B[k][j]
                    adra_el = ADRA + i * TM * TELEM + k * TELEM;
                    adrb_el = ADRB + k * TM * TELEM + jk * TELEM;

                    // cache access
                    if (pol == 0) {
                        NhitA += cache (adra_el, init_lru, maj_lru);
                        NhitB += cache (adrb_el, init_lru, maj_lru);
                    } else {
                        NhitB += cache (adra_el, init_lfu, maj_lfu);
                        NhitA += cache (adrb_el, init_lfu, maj_lfu);
                    }
                }
            }
        }
    }
```

## Step 3
```c
    int h = 16;
    for (ki = 0; ki < TM; ki += h) {
        for (i = 0; i < TM; ++ i) {
            for (j = 0; j < TM; ++ j) {
                int m = ki + 16;
                for (k = ki; k < m; ++ k) {

                    // A[i][k] * B[k][j]
                    adra_el = ADRA + i * TM * TELEM + k * TELEM;
                    adrb_el = ADRB + k * TM * TELEM + j * TELEM;

                    // cache access
                    if (pol == 0) {
                        NhitA += cache(adra_el, init_lru, maj_lru);
                        NhitB += cache(adrb_el, init_lru, maj_lru);
                    } else {
                        NhitB += cache(adra_el, init_lfu, maj_lfu);
                        NhitA += cache(adrb_el, init_lfu, maj_lfu);
                    }
                }
            }
        }
    }
```