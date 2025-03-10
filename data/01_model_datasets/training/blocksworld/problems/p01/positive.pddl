

(define (problem BW-rand-10)
(:domain blocksworld-4ops)
(:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 )
(:init
(arm-empty)
(on b1 b8)
(on b2 b5)
(on b3 b9)
(on-table b4)
(on-table b5)
(on-table b6)
(on b7 b3)
(on-table b8)
(on b9 b6)
(on b10 b1)
(clear b2)
(clear b4)
(clear b7)
(clear b10)
)
(:goal
(and
(on b2 b1)
(on b3 b7)
(on b4 b6)
(on b5 b8)
(on b6 b3)
(on b9 b10)
(on b10 b4))
)
)


