(define (problem BW-rand-10)
    (:domain blocksworld-4ops)
    (:objects b1 b10 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init (arm-empty) (clear b1) (clear b3) (clear b4) (clear b5) (clear b8) (on b3 b6) (on b4 b7) (on b6 b10) (on b7 b9) (on b9 b2) (on-table b1) (on-table b10) (on-table b2) (on-table b5) (on-table b8))
    (:goal (and (on b1 b2) (on b3 b5) (on b4 b10) (on b5 b7) (on b6 b4) (on b7 b6) (on b8 b1) (on b9 b3) (on b10 b8)))
)