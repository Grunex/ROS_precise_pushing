; Auto-generated. Do not edit!


(cl:in-package precise_pushing_bridge-srv)


;//! \htmlinclude Step-request.msg.html

(cl:defclass <Step-request> (roslisp-msg-protocol:ros-message)
  ((action
    :reader action
    :initarg :action
    :type (cl:vector cl:float)
   :initform (cl:make-array 0 :element-type 'cl:float :initial-element 0.0)))
)

(cl:defclass Step-request (<Step-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <Step-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'Step-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name precise_pushing_bridge-srv:<Step-request> is deprecated: use precise_pushing_bridge-srv:Step-request instead.")))

(cl:ensure-generic-function 'action-val :lambda-list '(m))
(cl:defmethod action-val ((m <Step-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader precise_pushing_bridge-srv:action-val is deprecated.  Use precise_pushing_bridge-srv:action instead.")
  (action m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <Step-request>) ostream)
  "Serializes a message object of type '<Step-request>"
  (cl:let ((__ros_arr_len (cl:length (cl:slot-value msg 'action))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_arr_len) ostream))
  (cl:map cl:nil #'(cl:lambda (ele) (cl:let ((bits (roslisp-utils:encode-single-float-bits ele)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)))
   (cl:slot-value msg 'action))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <Step-request>) istream)
  "Deserializes a message object of type '<Step-request>"
  (cl:let ((__ros_arr_len 0))
    (cl:setf (cl:ldb (cl:byte 8 0) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 16) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 24) __ros_arr_len) (cl:read-byte istream))
  (cl:setf (cl:slot-value msg 'action) (cl:make-array __ros_arr_len))
  (cl:let ((vals (cl:slot-value msg 'action)))
    (cl:dotimes (i __ros_arr_len)
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:aref vals i) (roslisp-utils:decode-single-float-bits bits))))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<Step-request>)))
  "Returns string type for a service object of type '<Step-request>"
  "precise_pushing_bridge/StepRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'Step-request)))
  "Returns string type for a service object of type 'Step-request"
  "precise_pushing_bridge/StepRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<Step-request>)))
  "Returns md5sum for a message object of type '<Step-request>"
  "8cc3aa28258e546257d5cfcc54ad965b")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'Step-request)))
  "Returns md5sum for a message object of type 'Step-request"
  "8cc3aa28258e546257d5cfcc54ad965b")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<Step-request>)))
  "Returns full string definition for message of type '<Step-request>"
  (cl:format cl:nil "# Request: action vector (shape must match your env's action_space)~%float32[] action~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'Step-request)))
  "Returns full string definition for message of type 'Step-request"
  (cl:format cl:nil "# Request: action vector (shape must match your env's action_space)~%float32[] action~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <Step-request>))
  (cl:+ 0
     4 (cl:reduce #'cl:+ (cl:slot-value msg 'action) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 4)))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <Step-request>))
  "Converts a ROS message object to a list"
  (cl:list 'Step-request
    (cl:cons ':action (action msg))
))
;//! \htmlinclude Step-response.msg.html

(cl:defclass <Step-response> (roslisp-msg-protocol:ros-message)
  ((reward
    :reader reward
    :initarg :reward
    :type cl:float
    :initform 0.0)
   (done
    :reader done
    :initarg :done
    :type cl:boolean
    :initform cl:nil))
)

(cl:defclass Step-response (<Step-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <Step-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'Step-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name precise_pushing_bridge-srv:<Step-response> is deprecated: use precise_pushing_bridge-srv:Step-response instead.")))

(cl:ensure-generic-function 'reward-val :lambda-list '(m))
(cl:defmethod reward-val ((m <Step-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader precise_pushing_bridge-srv:reward-val is deprecated.  Use precise_pushing_bridge-srv:reward instead.")
  (reward m))

(cl:ensure-generic-function 'done-val :lambda-list '(m))
(cl:defmethod done-val ((m <Step-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader precise_pushing_bridge-srv:done-val is deprecated.  Use precise_pushing_bridge-srv:done instead.")
  (done m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <Step-response>) ostream)
  "Serializes a message object of type '<Step-response>"
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'reward))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'done) 1 0)) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <Step-response>) istream)
  "Deserializes a message object of type '<Step-response>"
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'reward) (roslisp-utils:decode-single-float-bits bits)))
    (cl:setf (cl:slot-value msg 'done) (cl:not (cl:zerop (cl:read-byte istream))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<Step-response>)))
  "Returns string type for a service object of type '<Step-response>"
  "precise_pushing_bridge/StepResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'Step-response)))
  "Returns string type for a service object of type 'Step-response"
  "precise_pushing_bridge/StepResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<Step-response>)))
  "Returns md5sum for a message object of type '<Step-response>"
  "8cc3aa28258e546257d5cfcc54ad965b")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'Step-response)))
  "Returns md5sum for a message object of type 'Step-response"
  "8cc3aa28258e546257d5cfcc54ad965b")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<Step-response>)))
  "Returns full string definition for message of type '<Step-response>"
  (cl:format cl:nil "# Response~%float32 reward~%bool done~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'Step-response)))
  "Returns full string definition for message of type 'Step-response"
  (cl:format cl:nil "# Response~%float32 reward~%bool done~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <Step-response>))
  (cl:+ 0
     4
     1
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <Step-response>))
  "Converts a ROS message object to a list"
  (cl:list 'Step-response
    (cl:cons ':reward (reward msg))
    (cl:cons ':done (done msg))
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'Step)))
  'Step-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'Step)))
  'Step-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'Step)))
  "Returns string type for a service object of type '<Step>"
  "precise_pushing_bridge/Step")