(define (problem BW-rand-10)
    (:domain blocksworld-4ops)
    (:objects b1 b10 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init (arm-empty) (clear b1) (clear b2) (clear b6) (clear b8) (on b1 b4) (on b3 b7) (on b4 b3) (on b5 b9) (on b6 b5) (on b8 b10) (on-table b10) (on-table b2) (on-table b7) (on-table b9))
    (:goal (and (on b1 b8) (on b3 b2) (on b4 b7) (on b5 b10) (on b8 b3) (on b9 b1) (on b10 b4)))
)