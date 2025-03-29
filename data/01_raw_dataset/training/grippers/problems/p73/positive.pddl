(define (problem gripper-3-7-15)
    (:domain gripper-strips)
    (:objects lgripper1 lgripper2 lgripper3 rgripper1 rgripper2 rgripper3 - gripper ball1 ball10 ball11 ball12 ball13 ball14 ball15 ball2 ball3 ball4 ball5 ball6 ball7 ball8 ball9 - object robot1 robot2 robot3 - robot room1 room2 room3 room4 room5 room6 room7 - room)
    (:init (at ball1 room1) (at ball10 room4) (at ball11 room4) (at ball12 room4) (at ball13 room3) (at ball14 room7) (at ball15 room4) (at ball2 room1) (at ball3 room3) (at ball4 room6) (at ball5 room5) (at ball6 room4) (at ball7 room3) (at ball8 room5) (at ball9 room1) (at-robby robot1 room3) (at-robby robot2 room3) (at-robby robot3 room2) (free robot1 lgripper1) (free robot1 rgripper1) (free robot2 lgripper2) (free robot2 rgripper2) (free robot3 lgripper3) (free robot3 rgripper3))
    (:goal (and (at ball1 room1) (at ball2 room6) (at ball3 room1) (at ball4 room3) (at ball5 room6) (at ball6 room7) (at ball7 room3) (at ball8 room5) (at ball9 room5) (at ball10 room5) (at ball11 room3) (at ball12 room4) (at ball13 room5) (at ball14 room6) (at ball15 room7)))
)