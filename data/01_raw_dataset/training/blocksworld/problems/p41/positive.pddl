(define (problem BW-rand-10)
    (:domain blocksworld-4ops)
    (:objects b1 b10 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init (arm-empty) (clear b10) (clear b2) (clear b8) (clear b9) (on b1 b5) (on b10 b3) (on b2 b7) (on b3 b4) (on b4 b6) (on b7 b1) (on-table b5) (on-table b6) (on-table b8) (on-table b9))
    (:goal (and (on b1 b3) (on b2 b6) (on b3 b10) (on b4 b1) (on b5 b2) (on b6 b4) (on b8 b9) (on b9 b7)))
)