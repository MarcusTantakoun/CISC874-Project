(define (problem grid_4room2_fpl_s0_seed212_n0)
    (:domain grid)
    (:objects key0 key1 p0 p1 p10 p11 p12 p13 p14 p15 p16 p17 p18 p2 p3 p4 p5 p6 p7 p8 p9 shape0 shape1)
    (:init (arm-empty) (at key0 p1) (at key1 p14) (at-robot p18) (conn p0 p1) (conn p0 p5) (conn p1 p0) (conn p1 p2) (conn p1 p6) (conn p10 p13) (conn p10 p7) (conn p11 p12) (conn p11 p15) (conn p11 p9) (conn p12 p11) (conn p12 p16) (conn p13 p10) (conn p13 p14) (conn p13 p17) (conn p14 p13) (conn p14 p18) (conn p15 p11) (conn p15 p16) (conn p16 p12) (conn p16 p15) (conn p17 p13) (conn p17 p18) (conn p18 p14) (conn p18 p17) (conn p2 p1) (conn p2 p3) (conn p3 p2) (conn p3 p4) (conn p3 p7) (conn p4 p3) (conn p4 p8) (conn p5 p0) (conn p5 p6) (conn p5 p9) (conn p6 p1) (conn p6 p5) (conn p7 p10) (conn p7 p3) (conn p7 p8) (conn p8 p4) (conn p8 p7) (conn p9 p11) (conn p9 p5) (key key0) (key key1) (key-shape key0 shape0) (key-shape key1 shape1) (lock-shape p10 shape1) (lock-shape p2 shape0) (lock-shape p9 shape1) (locked p10) (locked p2) (locked p9) (open p0) (open p1) (open p11) (open p12) (open p13) (open p14) (open p15) (open p16) (open p17) (open p18) (open p3) (open p4) (open p5) (open p6) (open p7) (open p8) (place p0) (place p1) (place p10) (place p11) (place p12) (place p13) (place p14) (place p15) (place p16) (place p17) (place p18) (place p2) (place p3) (place p4) (place p5) (place p6) (place p7) (place p8) (place p9) (shape shape0) (shape shape1))
    (:goal (at-robot p7))
)