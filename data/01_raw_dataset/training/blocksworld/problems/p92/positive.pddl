(define (problem BW-rand-10)
    (:domain blocksworld-4ops)
    (:objects b1 b10 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init (arm-empty) (clear b5) (clear b8) (clear b9) (on b10 b2) (on b2 b6) (on b3 b10) (on b5 b7) (on b6 b1) (on b8 b3) (on b9 b4) (on-table b1) (on-table b4) (on-table b7))
    (:goal (and (on b3 b10) (on b5 b3) (on b6 b7) (on b7 b4) (on b8 b9)))
)