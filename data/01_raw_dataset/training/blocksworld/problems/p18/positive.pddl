(define (problem BW-rand-10)
    (:domain blocksworld-4ops)
    (:objects b1 b10 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init (arm-empty) (clear b2) (clear b3) (clear b4) (clear b5) (on b1 b7) (on b10 b6) (on b2 b10) (on b3 b1) (on b6 b9) (on b7 b8) (on-table b4) (on-table b5) (on-table b8) (on-table b9))
    (:goal (and (on b1 b7) (on b2 b10) (on b4 b9) (on b9 b3) (on b10 b4)))
)