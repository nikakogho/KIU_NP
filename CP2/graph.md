```
Legend:
  [X]    = state node (temperature variable)
  --->   = "upstream temperature feeds into ODE of downstream node" (advection / input)
  ==UA== = heat-transfer coupling term UA*(T_left - T_right) appears in BOTH ODEs
  ~rad~  = radiation/solar term depends on that node's own temperature (T^4 etc.)

------------------------ RADIATOR / SPACE ------------------------

                ~rad~ (depends on T_r)
                 |
               [T_r]
                 ^
                 | ==UA_r==
                 v
               [T_cR]  <---  T_mix  (mixing boundary for manifold advection)
                 ^
                 |
                 |  (this T_cR is upstream boundary for BOTH supply chains)
                 |
      ---------------------------------------------------------
      |                                                       |
      |                                                       |
  Branch A supply                                         Branch B supply
      |                                                       |
      v                                                       v
 [T_sup_A1] ---> [T_sup_A2] ---> [T_sup_A3] ---> [T_sup_A4] ---> [T_sup_A5] ---> (feeds cold plate A)
      ^                                                       ^
      |                                                       |
     ~rad~ on each segment                                   ~rad~ on each segment
      |                                                       |
      ---------------------------------------------------------

------------------------ SERVER / COLD PLATES ------------------------

Cold plate A coupling:
   [T_sup_A5]  ---> (inlet term in dT_cA/dt)
        |
        v
      [T_cA]  <==UA_sA==>  [T_sA]
        |
        v
   [T_ret_A1] ---> [T_ret_A2] ---> [T_ret_A3] ---> [T_ret_A4] ---> [T_ret_A5]
        ^                                                         ^
        |                                                         |
       ~rad~ on each segment                                     ~rad~ on each segment

Cold plate B coupling:
   [T_sup_B5]  ---> (inlet term in dT_cB/dt)
        |
        v
      [T_cB]  <==UA_sB==>  [T_sB]
        |
        v
   [T_ret_B1] ---> [T_ret_B2] ---> [T_ret_B3] ---> [T_ret_B4] ---> [T_ret_B5]
        ^                                                         ^
        |                                                         |
       ~rad~ on each segment                                     ~rad~ on each segment

------------------------ MIXING (MERGE) ------------------------

[T_ret_A5] ----\
                 \ 
                  >---- T_mix = (mdot_A*T_ret_A5 + mdot_B*T_ret_B5)/(mdot_A+mdot_B)
                 /
[T_ret_B5] ----/
                      |
                      v
                    [T_cR]  (advection term uses (T_mix - T_cR))

Notes:
- Each pipe segment's ODE depends on its upstream neighbor (advection) + its own ~rad~ term.
- Server solid ODE: depends on P(t), T_s, T_c.
- Cold plate coolant ODE: depends on T_s, T_c, and inlet T_sup_*5.
- Radiator manifold ODE: depends on T_mix, T_cR, and T_r.
- Radiator panel ODE: depends on T_cR and nonlinear radiation (T_r^4).
```