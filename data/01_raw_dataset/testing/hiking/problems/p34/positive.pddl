(define (problem Hiking-5-7-3)
    (:domain hiking)
    (:objects car0 car1 car2 car3 car4 car5 car6 - car couple0 couple1 couple2 couple3 couple4 - couple girl0 girl1 girl2 girl3 girl4 guy0 guy1 guy2 guy3 guy4 - person place0 place1 place2 - place tent0 tent1 tent2 tent3 tent4 - tent)
    (:init (at_car car0 place0) (at_car car1 place0) (at_car car2 place0) (at_car car3 place0) (at_car car4 place0) (at_car car5 place0) (at_car car6 place0) (at_person girl0 place0) (at_person girl1 place0) (at_person girl2 place0) (at_person girl3 place0) (at_person girl4 place0) (at_person guy0 place0) (at_person guy1 place0) (at_person guy2 place0) (at_person guy3 place0) (at_person guy4 place0) (at_tent tent0 place0) (at_tent tent1 place0) (at_tent tent2 place0) (at_tent tent3 place0) (at_tent tent4 place0) (down tent0) (next place0 place1) (next place1 place2) (partners couple0 guy0 girl0) (partners couple1 guy1 girl1) (partners couple2 guy2 girl2) (partners couple3 guy3 girl3) (partners couple4 guy4 girl4) (up tent1) (up tent2) (up tent3) (up tent4) (walked couple0 place0) (walked couple1 place0) (walked couple2 place0) (walked couple3 place0) (walked couple4 place0))
    (:goal (and (walked couple0 place2) (walked couple1 place2) (walked couple2 place2) (walked couple3 place2) (walked couple4 place2)))
)