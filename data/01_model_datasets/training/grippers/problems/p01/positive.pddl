(define (problem gripper-2-4-10)
(:domain gripper-strips)
(:objects robot1 robot2 - robot
rgripper1 lgripper1 rgripper2 lgripper2 - gripper
room1 room2 room3 room4 - room
ball1 ball2 ball3 ball4 ball5 ball6 ball7 ball8 ball9 ball10 - object)
(:init
(at-robby robot1 room4)
(free robot1 rgripper1)
(free robot1 lgripper1)
(at-robby robot2 room2)
(free robot2 rgripper2)
(free robot2 lgripper2)
(at ball1 room4)
(at ball2 room3)
(at ball3 room3)
(at ball4 room3)
(at ball5 room3)
(at ball6 room1)
(at ball7 room1)
(at ball8 room3)
(at ball9 room3)
(at ball10 room2)
)
(:goal
(and
(at ball1 room1)
(at ball2 room1)
(at ball3 room3)
(at ball4 room1)
(at ball5 room4)
(at ball6 room1)
(at ball7 room1)
(at ball8 room3)
(at ball9 room3)
(at ball10 room3)
)
)
)
