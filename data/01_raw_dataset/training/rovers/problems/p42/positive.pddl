(define (problem roverprob448)
    (:domain Rover)
    (:objects camera0 camera1 - Camera general - Lander colour high_res low_res - Mode objective0 objective1 - Objective rover0 rover1 - Rover rover0store rover1store - Store waypoint0 waypoint1 waypoint2 waypoint3 - Waypoint)
    (:init (at rover0 waypoint1) (at rover1 waypoint3) (at_lander general waypoint1) (at_rock_sample waypoint0) (at_rock_sample waypoint1) (at_rock_sample waypoint3) (at_soil_sample waypoint0) (at_soil_sample waypoint1) (at_soil_sample waypoint3) (available rover0) (available rover1) (calibration_target camera0 objective0) (calibration_target camera1 objective1) (can_traverse rover0 waypoint0 waypoint1) (can_traverse rover0 waypoint1 waypoint0) (can_traverse rover0 waypoint1 waypoint3) (can_traverse rover0 waypoint3 waypoint1) (can_traverse rover1 waypoint0 waypoint1) (can_traverse rover1 waypoint0 waypoint2) (can_traverse rover1 waypoint0 waypoint3) (can_traverse rover1 waypoint1 waypoint0) (can_traverse rover1 waypoint2 waypoint0) (can_traverse rover1 waypoint3 waypoint0) (channel_free general) (empty rover0store) (empty rover1store) (equipped_for_imaging rover0) (equipped_for_soil_analysis rover1) (on_board camera0 rover0) (on_board camera1 rover0) (store_of rover0store rover0) (store_of rover1store rover1) (supports camera0 high_res) (supports camera0 low_res) (supports camera1 high_res) (visible waypoint0 waypoint1) (visible waypoint0 waypoint2) (visible waypoint0 waypoint3) (visible waypoint1 waypoint0) (visible waypoint1 waypoint2) (visible waypoint1 waypoint3) (visible waypoint2 waypoint0) (visible waypoint2 waypoint1) (visible waypoint2 waypoint3) (visible waypoint3 waypoint0) (visible waypoint3 waypoint1) (visible waypoint3 waypoint2) (visible_from objective0 waypoint0) (visible_from objective0 waypoint1) (visible_from objective1 waypoint2) (visible_from objective1 waypoint3))
    (:goal (and (communicated_soil_data waypoint3) (communicated_image_data objective0 high_res)))
)