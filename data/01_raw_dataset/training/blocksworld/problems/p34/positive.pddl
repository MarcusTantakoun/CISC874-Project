(define (problem BW-rand-10)
    (:domain blocksworld-4ops)
    (:objects b1 b10 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init (arm-empty) (clear b10) (clear b5) (clear b7) (on b10 b9) (on b2 b3) (on b3 b1) (on b4 b8) (on b5 b6) (on b6 b4) (on b7 b2) (on-table b1) (on-table b8) (on-table b9))
    (:goal (and (on b1 b3) (on b2 b7) (on b3 b2) (on b4 b1) (on b5 b10) (on b6 b8) (on b7 b9) (on b9 b6) (on b10 b4)))
)