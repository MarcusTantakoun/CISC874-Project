(define (problem gripper-4-3-11)
    (:domain gripper-strips)
    (:objects lgripper1 lgripper2 lgripper3 lgripper4 rgripper1 rgripper2 rgripper3 rgripper4 - gripper ball1 ball10 ball11 ball2 ball3 ball4 ball5 ball6 ball7 ball8 ball9 - object robot1 robot2 robot3 robot4 - robot room1 room2 room3 - room)
    (:init (at ball1 room3) (at ball10 room3) (at ball11 room2) (at ball2 room2) (at ball3 room2) (at ball4 room2) (at ball5 room2) (at ball6 room3) (at ball7 room2) (at ball8 room3) (at ball9 room1) (at-robby robot1 room1) (at-robby robot2 room2) (at-robby robot3 room1) (at-robby robot4 room3) (free robot1 lgripper1) (free robot1 rgripper1) (free robot2 lgripper2) (free robot2 rgripper2) (free robot3 lgripper3) (free robot3 rgripper3) (free robot4 lgripper4) (free robot4 rgripper4))
    (:goal (and (at ball1 room3) (at ball2 room2) (at ball3 room2) (at ball4 room3) (at ball5 room3) (at ball6 room2) (at ball7 room3) (at ball8 room1) (at ball9 room1) (at ball10 room2) (at ball11 room2)))
)