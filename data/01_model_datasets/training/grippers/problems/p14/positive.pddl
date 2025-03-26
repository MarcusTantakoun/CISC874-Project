(define (problem gripper-4-4-13)
    (:domain gripper-strips)
    (:objects lgripper1 lgripper2 lgripper3 lgripper4 rgripper1 rgripper2 rgripper3 rgripper4 - gripper ball1 ball10 ball11 ball12 ball13 ball2 ball3 ball4 ball5 ball6 ball7 ball8 ball9 - object robot1 robot2 robot3 robot4 - robot room1 room2 room3 room4 - room)
    (:init (at ball1 room1) (at ball10 room2) (at ball11 room3) (at ball12 room4) (at ball13 room4) (at ball2 room4) (at ball3 room1) (at ball4 room4) (at ball5 room4) (at ball6 room2) (at ball7 room1) (at ball8 room1) (at ball9 room1) (at-robby robot1 room4) (at-robby robot2 room4) (at-robby robot3 room2) (at-robby robot4 room3) (free robot1 lgripper1) (free robot1 rgripper1) (free robot2 lgripper2) (free robot2 rgripper2) (free robot3 lgripper3) (free robot3 rgripper3) (free robot4 lgripper4) (free robot4 rgripper4))
    (:goal (and (at ball1 room2) (at ball2 room4) (at ball3 room2) (at ball4 room2) (at ball5 room3) (at ball6 room1) (at ball7 room1) (at ball8 room4) (at ball9 room3) (at ball10 room4) (at ball11 room3) (at ball12 room3) (at ball13 room2)))
)