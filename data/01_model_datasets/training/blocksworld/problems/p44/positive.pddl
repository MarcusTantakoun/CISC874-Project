(define (problem BW-rand-10)
    (:domain blocksworld-4ops)
    (:objects b1 b10 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init (arm-empty) (clear b10) (clear b2) (clear b9) (on b1 b3) (on b10 b1) (on b2 b8) (on b6 b4) (on b7 b6) (on b8 b7) (on b9 b5) (on-table b3) (on-table b4) (on-table b5))
    (:goal (and (on b1 b9) (on b2 b10) (on b3 b8) (on b4 b3) (on b5 b1) (on b6 b7) (on b10 b6)))
)