(define (problem BW-rand-10)
    (:domain blocksworld-4ops)
    (:objects b1 b10 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init (arm-empty) (clear b6) (clear b7) (on b1 b9) (on b10 b8) (on b2 b3) (on b3 b5) (on b4 b1) (on b5 b4) (on b7 b10) (on b8 b2) (on-table b6) (on-table b9))
    (:goal (and (on b1 b2) (on b4 b6) (on b6 b9) (on b7 b10) (on b9 b3) (on b10 b4)))
)