(define (problem BW-rand-10)
    (:domain blocksworld-4ops)
    (:objects b1 b10 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init (arm-empty) (clear b3) (clear b5) (clear b6) (clear b9) (on b1 b4) (on b3 b8) (on b4 b10) (on b5 b1) (on b6 b7) (on b7 b2) (on-table b10) (on-table b2) (on-table b8) (on-table b9))
    (:goal (and (on b1 b8) (on b2 b9) (on b3 b10) (on b4 b5) (on b5 b2) (on b7 b3) (on b9 b6) (on b10 b1)))
)