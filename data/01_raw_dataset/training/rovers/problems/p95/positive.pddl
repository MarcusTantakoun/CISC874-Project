(define (problem roverprob623)
    (:domain Rover)
    (:objects camera0 camera1 - Camera general - Lander colour high_res low_res - Mode objective0 objective1 - Objective rover0 rover1 - Rover rover0store rover1store - Store waypoint0 waypoint1 waypoint2 waypoint3 - Waypoint)
    (:init (at rover0 waypoint1) (at rover1 waypoint3) (at_lander general waypoint2) (available rover0) (available rover1) (calibration_target camera0 objective1) (calibration_target camera1 objective0) (can_traverse rover0 waypoint0 waypoint1) (can_traverse rover0 waypoint0 waypoint2) (can_traverse rover0 waypoint1 waypoint0) (can_traverse rover0 waypoint1 waypoint3) (can_traverse rover0 waypoint2 waypoint0) (can_traverse rover0 waypoint3 waypoint1) (can_traverse rover1 waypoint0 waypoint2) (can_traverse rover1 waypoint0 waypoint3) (can_traverse rover1 waypoint1 waypoint3) (can_traverse rover1 waypoint2 waypoint0) (can_traverse rover1 waypoint3 waypoint0) (can_traverse rover1 waypoint3 waypoint1) (channel_free general) (empty rover0store) (empty rover1store) (equipped_for_imaging rover0) (equipped_for_imaging rover1) (equipped_for_rock_analysis rover1) (equipped_for_soil_analysis rover1) (on_board camera0 rover0) (on_board camera1 rover1) (store_of rover0store rover0) (store_of rover1store rover1) (supports camera0 low_res) (supports camera1 high_res) (supports camera1 low_res) (visible waypoint0 waypoint1) (visible waypoint0 waypoint2) (visible waypoint0 waypoint3) (visible waypoint1 waypoint0) (visible waypoint1 waypoint2) (visible waypoint1 waypoint3) (visible waypoint2 waypoint0) (visible waypoint2 waypoint1) (visible waypoint2 waypoint3) (visible waypoint3 waypoint0) (visible waypoint3 waypoint1) (visible waypoint3 waypoint2) (visible_from objective0 waypoint1) (visible_from objective0 waypoint2) (visible_from objective0 waypoint3) (visible_from objective1 waypoint3))
    (:goal (and (communicated_image_data objective0 low_res) (communicated_image_data objective1 low_res)))
)