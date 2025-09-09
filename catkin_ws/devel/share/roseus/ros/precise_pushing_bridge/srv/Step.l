;; Auto-generated. Do not edit!


(when (boundp 'precise_pushing_bridge::Step)
  (if (not (find-package "PRECISE_PUSHING_BRIDGE"))
    (make-package "PRECISE_PUSHING_BRIDGE"))
  (shadow 'Step (find-package "PRECISE_PUSHING_BRIDGE")))
(unless (find-package "PRECISE_PUSHING_BRIDGE::STEP")
  (make-package "PRECISE_PUSHING_BRIDGE::STEP"))
(unless (find-package "PRECISE_PUSHING_BRIDGE::STEPREQUEST")
  (make-package "PRECISE_PUSHING_BRIDGE::STEPREQUEST"))
(unless (find-package "PRECISE_PUSHING_BRIDGE::STEPRESPONSE")
  (make-package "PRECISE_PUSHING_BRIDGE::STEPRESPONSE"))

(in-package "ROS")





(defclass precise_pushing_bridge::StepRequest
  :super ros::object
  :slots (_action ))

(defmethod precise_pushing_bridge::StepRequest
  (:init
   (&key
    ((:action __action) (make-array 0 :initial-element 0.0 :element-type :float))
    )
   (send-super :init)
   (setq _action __action)
   self)
  (:action
   (&optional __action)
   (if __action (setq _action __action)) _action)
  (:serialization-length
   ()
   (+
    ;; float32[] _action
    (* 4    (length _action)) 4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; float32[] _action
     (write-long (length _action) s)
     (dotimes (i (length _action))
       (sys::poke (elt _action i) (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
       )
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; float32[] _action
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _action (instantiate float-vector n))
     (dotimes (i n)
     (setf (elt _action i) (sys::peek buf ptr- :float)) (incf ptr- 4)
     ))
   ;;
   self)
  )

(defclass precise_pushing_bridge::StepResponse
  :super ros::object
  :slots (_reward _done ))

(defmethod precise_pushing_bridge::StepResponse
  (:init
   (&key
    ((:reward __reward) 0.0)
    ((:done __done) nil)
    )
   (send-super :init)
   (setq _reward (float __reward))
   (setq _done __done)
   self)
  (:reward
   (&optional __reward)
   (if __reward (setq _reward __reward)) _reward)
  (:done
   (&optional (__done :null))
   (if (not (eq __done :null)) (setq _done __done)) _done)
  (:serialization-length
   ()
   (+
    ;; float32 _reward
    4
    ;; bool _done
    1
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; float32 _reward
       (sys::poke _reward (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; bool _done
       (if _done (write-byte -1 s) (write-byte 0 s))
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; float32 _reward
     (setq _reward (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; bool _done
     (setq _done (not (= 0 (sys::peek buf ptr- :char)))) (incf ptr- 1)
   ;;
   self)
  )

(defclass precise_pushing_bridge::Step
  :super ros::object
  :slots ())

(setf (get precise_pushing_bridge::Step :md5sum-) "8cc3aa28258e546257d5cfcc54ad965b")
(setf (get precise_pushing_bridge::Step :datatype-) "precise_pushing_bridge/Step")
(setf (get precise_pushing_bridge::Step :request) precise_pushing_bridge::StepRequest)
(setf (get precise_pushing_bridge::Step :response) precise_pushing_bridge::StepResponse)

(defmethod precise_pushing_bridge::StepRequest
  (:response () (instance precise_pushing_bridge::StepResponse :init)))

(setf (get precise_pushing_bridge::StepRequest :md5sum-) "8cc3aa28258e546257d5cfcc54ad965b")
(setf (get precise_pushing_bridge::StepRequest :datatype-) "precise_pushing_bridge/StepRequest")
(setf (get precise_pushing_bridge::StepRequest :definition-)
      "# Request: action vector (shape must match your env's action_space)
float32[] action
---
# Response
float32 reward
bool done

")

(setf (get precise_pushing_bridge::StepResponse :md5sum-) "8cc3aa28258e546257d5cfcc54ad965b")
(setf (get precise_pushing_bridge::StepResponse :datatype-) "precise_pushing_bridge/StepResponse")
(setf (get precise_pushing_bridge::StepResponse :definition-)
      "# Request: action vector (shape must match your env's action_space)
float32[] action
---
# Response
float32 reward
bool done

")



(provide :precise_pushing_bridge/Step "8cc3aa28258e546257d5cfcc54ad965b")


