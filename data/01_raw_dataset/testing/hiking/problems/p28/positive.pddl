(define (problem Hiking-2-6-2)
    (:domain hiking)
    (:objects car0 car1 car2 car3 car4 car5 - car couple0 couple1 - couple girl0 girl1 guy0 guy1 - person place0 place1 - place tent0 tent1 - tent)
    (:init (at_car car0 place0) (at_car car1 place0) (at_car car2 place0) (at_car car3 place0) (at_car car4 place0) (at_car car5 place0) (at_person girl0 place0) (at_person girl1 place0) (at_person guy0 place0) (at_person guy1 place0) (at_tent tent0 place0) (at_tent tent1 place0) (down tent0) (next place0 place1) (partners couple0 guy0 girl0) (partners couple1 guy1 girl1) (up tent1) (walked couple0 place0) (walked couple1 place0))
    (:goal (and (walked couple0 place1) (walked couple1 place1)))
)