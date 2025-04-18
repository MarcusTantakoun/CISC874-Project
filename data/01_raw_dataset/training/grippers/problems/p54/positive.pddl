(define (problem gripper-3-6-3)
    (:domain gripper-strips)
    (:objects lgripper1 lgripper2 lgripper3 rgripper1 rgripper2 rgripper3 - gripper ball1 ball2 ball3 - object robot1 robot2 robot3 - robot room1 room2 room3 room4 room5 room6 - room)
    (:init (at ball1 room2) (at ball2 room5) (at ball3 room4) (at-robby robot1 room1) (at-robby robot2 room6) (at-robby robot3 room3) (free robot1 lgripper1) (free robot1 rgripper1) (free robot2 lgripper2) (free robot2 rgripper2) (free robot3 lgripper3) (free robot3 rgripper3))
    (:goal (and (at ball1 room6) (at ball2 room3) (at ball3 room2)))
)