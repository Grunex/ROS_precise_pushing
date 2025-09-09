// Auto-generated. Do not edit!

// (in-package precise_pushing_bridge.srv)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;

//-----------------------------------------------------------


//-----------------------------------------------------------

class StepRequest {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.action = null;
    }
    else {
      if (initObj.hasOwnProperty('action')) {
        this.action = initObj.action
      }
      else {
        this.action = [];
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type StepRequest
    // Serialize message field [action]
    bufferOffset = _arraySerializer.float32(obj.action, buffer, bufferOffset, null);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type StepRequest
    let len;
    let data = new StepRequest(null);
    // Deserialize message field [action]
    data.action = _arrayDeserializer.float32(buffer, bufferOffset, null)
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += 4 * object.action.length;
    return length + 4;
  }

  static datatype() {
    // Returns string type for a service object
    return 'precise_pushing_bridge/StepRequest';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return 'a70a7d92e0376dcb967914076f276ea6';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    # Request: action vector (shape must match your env's action_space)
    float32[] action
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new StepRequest(null);
    if (msg.action !== undefined) {
      resolved.action = msg.action;
    }
    else {
      resolved.action = []
    }

    return resolved;
    }
};

class StepResponse {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.reward = null;
      this.done = null;
    }
    else {
      if (initObj.hasOwnProperty('reward')) {
        this.reward = initObj.reward
      }
      else {
        this.reward = 0.0;
      }
      if (initObj.hasOwnProperty('done')) {
        this.done = initObj.done
      }
      else {
        this.done = false;
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type StepResponse
    // Serialize message field [reward]
    bufferOffset = _serializer.float32(obj.reward, buffer, bufferOffset);
    // Serialize message field [done]
    bufferOffset = _serializer.bool(obj.done, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type StepResponse
    let len;
    let data = new StepResponse(null);
    // Deserialize message field [reward]
    data.reward = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [done]
    data.done = _deserializer.bool(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    return 5;
  }

  static datatype() {
    // Returns string type for a service object
    return 'precise_pushing_bridge/StepResponse';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '4a28b027ac30cbc047f58b69ea622b8a';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    # Response
    float32 reward
    bool done
    
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new StepResponse(null);
    if (msg.reward !== undefined) {
      resolved.reward = msg.reward;
    }
    else {
      resolved.reward = 0.0
    }

    if (msg.done !== undefined) {
      resolved.done = msg.done;
    }
    else {
      resolved.done = false
    }

    return resolved;
    }
};

module.exports = {
  Request: StepRequest,
  Response: StepResponse,
  md5sum() { return '8cc3aa28258e546257d5cfcc54ad965b'; },
  datatype() { return 'precise_pushing_bridge/Step'; }
};
