(define (problem Hiking-4-5-2)
    (:domain hiking)
    (:objects car0 car1 car2 car3 car4 - car couple0 couple1 couple2 couple3 - couple girl0 girl1 girl2 girl3 guy0 guy1 guy2 guy3 - person place0 place1 - place tent0 tent1 tent2 tent3 - tent)
    (:init (at_car car0 place0) (at_car car1 place0) (at_car car2 place0) (at_car car3 place0) (at_car car4 place0) (at_person girl0 place0) (at_person girl1 place0) (at_person girl2 place0) (at_person girl3 place0) (at_person guy0 place0) (at_person guy1 place0) (at_person guy2 place0) (at_person guy3 place0) (at_tent tent0 place0) (at_tent tent1 place0) (at_tent tent2 place0) (at_tent tent3 place0) (down tent1) (down tent3) (next place0 place1) (partners couple0 guy0 girl0) (partners couple1 guy1 girl1) (partners couple2 guy2 girl2) (partners couple3 guy3 girl3) (up tent0) (up tent2) (walked couple0 place0) (walked couple1 place0) (walked couple2 place0) (walked couple3 place0))
    (:goal (and (walked couple0 place1) (walked couple1 place1) (walked couple2 place1) (walked couple3 place1)))
)