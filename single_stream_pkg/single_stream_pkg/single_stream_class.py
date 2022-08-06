################################################################################
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
################################################################################

# This node performs detection and classification inference on a single input stream and publishes results to topics infer_detection and infer_classification respectively

# Required ROS2 imports
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from vision_msgs.msg import Classification2D, ObjectHypothesis, ObjectHypothesisWithPose, BoundingBox2D, Detection2D, Detection2DArray

import os
import sys
sys.path.append('/opt/nvidia/deepstream/deepstream/lib')
import platform
import configparser

import gi
gi.require_version('Gst', '1.0')
from gi.repository import GObject, Gst

import pyds

sys.path.insert(0, './src/ros2_deepstream')
from common.is_aarch_64 import is_aarch64
from common.bus_call import bus_call
from common.FPS import GETFPS

PGIE_CLASS_ID_PROHIBITORY = 0
PGIE_CLASS_ID_DANGER = 1
PGIE_CLASS_ID_MANDATORY = 2
PGIE_CLASS_ID_OTHER = 3

location = os.getcwd() + "/src/ros2_deepstream/config_files/"
# pgie class labels
class_sign_pgie = (open(location+'signs.names.txt').readline().rstrip('\n')).split(';')
# sgie class labels
class_sign_sgie = (open(location+'signs.sgie_classes.txt').readline().rstrip('\n')).split(';')

class InferencePublisher(Node):
    def osd_sink_pad_buffer_probe(self,pad,info,u_data):
        frame_number=0
        #Intializing object counter with 0.
        obj_counter = {
            PGIE_CLASS_ID_PROHIBITORY:0,
            PGIE_CLASS_ID_DANGER:0,
            PGIE_CLASS_ID_MANDATORY:0,
            PGIE_CLASS_ID_OTHER:0
        }


        num_rects=0

        gst_buffer = info.get_buffer()
        if not gst_buffer:
            print("Unable to get GstBuffer ")
            return

        # Retrieve batch metadata from the gst_buffer
        # Note that pyds.gst_buffer_get_nvds_batch_meta() expects the
        # C address of gst_buffer as input, which is obtained with hash(gst_buffer)
        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        l_frame = batch_meta.frame_meta_list
        while l_frame is not None:
            try:
                # Note that l_frame.data needs a cast to pyds.NvDsFrameMeta
                # The casting is done by pyds.NvDsFrameMeta.cast()
                # The casting also keeps ownership of the underlying memory
                # in the C code, so the Python garbage collector will leave
                # it alone.
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            except StopIteration:
                break

            frame_number=frame_meta.frame_num
            num_rects = frame_meta.num_obj_meta
            l_obj=frame_meta.obj_meta_list

            # Message for output of detection inference
            msg = Detection2DArray()
            while l_obj is not None:
                try:
                    # Casting l_obj.data to pyds.NvDsObjectMeta
                    obj_meta=pyds.NvDsObjectMeta.cast(l_obj.data)
                    l_classifier = obj_meta.classifier_meta_list

#                    # If object is a car (class ID 0), perform attribute classification
#                    if obj_meta.class_id == 0 and l_classifier is not None:
                    # perform attribute classification on ALL classes
                    if l_classifier is not None:
                        # Creating and publishing message with output of classification inference
                        msg2 = Classification2D()

                        while l_classifier is not None:
                            result = ObjectHypothesis()
                            try:
                                classifier_meta = pyds.glist_get_nvds_classifier_meta(l_classifier.data)
                                
                            except StopIteration:
                                print('Could not parse MetaData: ')
                                break

                            classifier_id = classifier_meta.unique_component_id
                            l_label = classifier_meta.label_info_list
                            label_info = pyds.glist_get_nvds_label_info(l_label.data)
                            classifier_class = label_info.result_class_id

#                            if classifier_id == 2:
#                                result.id = class_color[classifier_class]
#                            elif classifier_id == 3:
#                                result.id = class_make[classifier_class]
#                            else:
#                                result.id = class_type[classifier_class]
                            result.id = class_sign_sgie[classifier_class]

                            result.score = label_info.result_prob                            
                            msg2.results.append(result)
                            l_classifier = l_classifier.next
                    
                        self.publisher_classification.publish(msg2)
                except StopIteration:
                    break
    
                obj_counter[obj_meta.class_id] += 1

                # Creating message for output of detection inference
                result = ObjectHypothesisWithPose()
                result.id = str(class_sign_pgie[obj_meta.class_id])
                result.score = obj_meta.confidence
                
                left = obj_meta.rect_params.left
                top = obj_meta.rect_params.top
                width = obj_meta.rect_params.width
                height = obj_meta.rect_params.height
                bounding_box = BoundingBox2D()
                bounding_box.center.x = float(left + (width/2)) 
                bounding_box.center.y = float(top + (height/2))
                bounding_box.size_x = width
                bounding_box.size_y = height
                
                detection = Detection2D()
                detection.results.append(result)
                detection.bbox = bounding_box
                msg.detections.append(detection)

                try: 
                    l_obj=l_obj.next
                except StopIteration:
                    break

            # Publishing message with output of detection inference
            self.publisher_detection.publish(msg)
        

            # Acquiring a display meta object. The memory ownership remains in
            # the C code so downstream plugins can still access it. Otherwise
            # the garbage collector will claim it when this probe function exits.
            display_meta=pyds.nvds_acquire_display_meta_from_pool(batch_meta)
            display_meta.num_labels = 1
            py_nvosd_text_params = display_meta.text_params[0]
            # Setting display text to be shown on screen
            # Note that the pyds module allocates a buffer for the string, and the
            # memory will not be claimed by the garbage collector.
            # Reading the display_text field here will return the C address of the
            # allocated string. Use pyds.get_string() to get the string content.
            py_nvosd_text_params.display_text = "Frame Number={} Number of Objects={} Prohibitory_count={} Danger_count={} Mandatory_count={} Other_count={}".format(frame_number, num_rects, obj_counter[PGIE_CLASS_ID_PROHIBITORY], obj_counter[PGIE_CLASS_ID_DANGER], obj_counter[PGIE_CLASS_ID_MANDATORY], obj_counter[PGIE_CLASS_ID_OTHER])

            # Now set the offsets where the string should appear
            py_nvosd_text_params.x_offset = 10
            py_nvosd_text_params.y_offset = 12

            # Font , font-color and font-size
            py_nvosd_text_params.font_params.font_name = "Serif"
            py_nvosd_text_params.font_params.font_size = 10
            # set(red, green, blue, alpha); set to White
            py_nvosd_text_params.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)

            # Text background color
            py_nvosd_text_params.set_bg_clr = 1
            # set(red, green, blue, alpha); set to Black
            py_nvosd_text_params.text_bg_clr.set(0.0, 0.0, 0.0, 1.0)
            # Using pyds.get_string() to get display_text as string
            pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)
            try:
                l_frame=l_frame.next
            except StopIteration:
                break
			
        return Gst.PadProbeReturn.OK 


    def __init__(self):
        super().__init__('inference_publisher')
#        # Taking name of input source from user
#        self.declare_parameter('input_source')
#        param_ip_src = self.get_parameter('input_source').value
        
        self.publisher_detection = self.create_publisher(Detection2DArray, 'infer_detection', 10)
        self.publisher_classification = self.create_publisher(Classification2D, 'infer_classification', 10)

        # Standard GStreamer initialization
        GObject.threads_init()
        Gst.init(None)

        # Create gstreamer elements
        # Create Pipeline element that will form a connection of other elements
        print("Creating Pipeline \n ")
        self.pipeline = Gst.Pipeline()
        if not self.pipeline:
            sys.stderr.write(" Unable to create Pipeline \n")


        print("Creating Source \n ")        
#        source = Gst.ElementFactory.make("v4l2src", "usb-cam-source")
#        if not source:
#            sys.stderr.write(" Unable to create Source \n")
        source = Gst.ElementFactory.make("rosimagesrc", "ros-image-source")
        if not source:
            sys.stderr.write(" Unable to create Source \n")

        caps_v4l2src = Gst.ElementFactory.make("capsfilter", "v4l2src_caps")
        if not caps_v4l2src:
            sys.stderr.write(" Unable to create v4l2src capsfilter \n")


        print("Creating Video Converter \n")

        # videoconvert to make sure a superset of raw formats are supported
        vidconvsrc = Gst.ElementFactory.make("videoconvert", "convertor_src1")
        if not vidconvsrc:
            sys.stderr.write(" Unable to create videoconvert \n")

        # nvvideoconvert to convert incoming raw buffers to NVMM Mem (NvBufSurface API)
        nvvidconvsrc = Gst.ElementFactory.make("nvvideoconvert", "convertor_src2")
        if not nvvidconvsrc:
            sys.stderr.write(" Unable to create Nvvideoconvert \n")

        caps_vidconvsrc = Gst.ElementFactory.make("capsfilter", "nvmm_caps")
        if not caps_vidconvsrc:
            sys.stderr.write(" Unable to create capsfilter \n")

        # Create nvstreammux instance to form batches from one or more sources.
        streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
        if not streammux:
            sys.stderr.write(" Unable to create NvStreamMux \n")

        # Use nvinfer to run inferencing on decoder's output,
        # behaviour of inferencing is set through config file
        pgie = Gst.ElementFactory.make("nvinfer", "primary-inference1")
        if not pgie:
            sys.stderr.write(" Unable to create pgie1 \n")

        tracker = Gst.ElementFactory.make("nvtracker", "tracker")
        if not tracker:
            sys.stderr.write(" Unable to create tracker \n")

        sgie1 = Gst.ElementFactory.make("nvinfer", "secondary1-nvinference-engine")
        if not sgie1:
            sys.stderr.write(" Unable to make sgie1 \n")

#        sgie2 = Gst.ElementFactory.make("nvinfer", "secondary2-nvinference-engine")
#        if not sgie2:
#            sys.stderr.write(" Unable to make sgie2 \n")
#
#        sgie3 = Gst.ElementFactory.make("nvinfer", "secondary3-nvinference-engine")
#        if not sgie3:
#            sys.stderr.write(" Unable to make sgie3 \n")

        pgie2 = Gst.ElementFactory.make("nvinfer", "primary-inference2")
        if not pgie2:
            sys.stderr.write(" Unable to create pgie2 \n")

        nvvidconv1 = Gst.ElementFactory.make("nvvideoconvert", "convertor1")
        if not nvvidconv1:
            sys.stderr.write(" Unable to create nvvidconv1 \n")

        nvvidconv2 = Gst.ElementFactory.make("nvvideoconvert", "convertor2")
        if not nvvidconv2:
            sys.stderr.write(" Unable to create nvvidconv2 \n")

        # Create OSD to draw on the converted RGBA buffer
        nvosd1 = Gst.ElementFactory.make("nvdsosd", "onscreendisplay1")
        if not nvosd1:
            sys.stderr.write(" Unable to create nvosd1 \n")

        nvosd2 = Gst.ElementFactory.make("nvdsosd", "onscreendisplay2")
        if not nvosd2:
            sys.stderr.write(" Unable to create nvosd2 \n")

        # Finally render the osd output
        if is_aarch64():
            transform1 = Gst.ElementFactory.make("nvegltransform", "nvegl-transform1")
            transform2 = Gst.ElementFactory.make("nvegltransform", "nvegl-transform2")

        print("Creating EGLSink \n")
        sink1 = Gst.ElementFactory.make("nveglglessink", "nvvideo-renderer1")
        if not sink1:
            sys.stderr.write(" Unable to create egl sink1 \n")

        sink2 = Gst.ElementFactory.make("nveglglessink", "nvvideo-renderer2")
        if not sink2:
            sys.stderr.write(" Unable to create egl sink2 \n")

        
#        source.set_property('device', param_ip_src)
#        caps_v4l2src.set_property('caps', Gst.Caps.from_string("video/x-raw, framerate=30/1"))
        source.set_property('ros-topic', "/camera/color/image_raw")
        caps_v4l2src.set_property('caps', Gst.Caps.from_string("video/x-raw, framerate=30/1"))
        caps_vidconvsrc.set_property('caps', Gst.Caps.from_string("video/x-raw(memory:NVMM)"))
#        streammux.set_property('width', 1920)
#        streammux.set_property('height', 1080)
        streammux.set_property('width', 640)
        streammux.set_property('height', 480)
        streammux.set_property('batch-size', 1)
        streammux.set_property('batched-push-timeout', 4000000)

        #Set properties of pgie and sgie
        location = os.getcwd() + "/src/ros2_deepstream/config_files/"
        pgie.set_property('config-file-path', location+"dstest2_pgie_config.txt")
        sgie1.set_property('config-file-path', location+"dstest2_sgie1_config.txt")
        #sgie2.set_property('config-file-path', location+"dstest2_sgie2_config.txt")
        #sgie3.set_property('config-file-path', location+"dstest2_sgie3_config.txt")
        pgie2.set_property('config-file-path', location+"dstest1_pgie_config.txt")
        sink1.set_property('sync', False)
        sink2.set_property('sync', False)

        #Set properties of tracker
        config = configparser.ConfigParser()
        config.read(location+'dstest2_tracker_config.txt')
        config.sections()

        for key in config['tracker']:
            if key == 'tracker-width' :
                tracker_width = config.getint('tracker', key)
                tracker.set_property('tracker-width', tracker_width)
            if key == 'tracker-height' :
                tracker_height = config.getint('tracker', key)
                tracker.set_property('tracker-height', tracker_height)
            if key == 'gpu-id' :
                tracker_gpu_id = config.getint('tracker', key)
                tracker.set_property('gpu_id', tracker_gpu_id)
            if key == 'll-lib-file' :
                tracker_ll_lib_file = config.get('tracker', key)
                tracker.set_property('ll-lib-file', tracker_ll_lib_file)
            if key == 'll-config-file' :
                tracker_ll_config_file = config.get('tracker', key)
                tracker.set_property('ll-config-file', tracker_ll_config_file)
            if key == 'enable-batch-process' :
                tracker_enable_batch_process = config.getint('tracker', key)
                tracker.set_property('enable_batch_process', tracker_enable_batch_process)

        tee = Gst.ElementFactory.make('tee', 'tee')
        queue1 = Gst.ElementFactory.make('queue','infer1')
        queue2 = Gst.ElementFactory.make('queue','infer2')

        print("Adding elements to Pipeline \n")
        self.pipeline.add(source)
        self.pipeline.add(caps_v4l2src)
        self.pipeline.add(vidconvsrc)
        self.pipeline.add(nvvidconvsrc)
        self.pipeline.add(caps_vidconvsrc)
        self.pipeline.add(streammux)
        self.pipeline.add(pgie)
        self.pipeline.add(pgie2)
        self.pipeline.add(tracker)
        self.pipeline.add(sgie1)
        #self.pipeline.add(sgie2)
        #self.pipeline.add(sgie3)
        self.pipeline.add(nvvidconv1)
        self.pipeline.add(nvvidconv2)
        self.pipeline.add(nvosd1)
        self.pipeline.add(nvosd2)
        self.pipeline.add(sink1)
        self.pipeline.add(sink2)
        self.pipeline.add(tee)
        self.pipeline.add(queue1)
        self.pipeline.add(queue2)

        if is_aarch64():
            self.pipeline.add(transform1)
            self.pipeline.add(transform2)

        # Link the elements together
        print("Linking elements in the Pipeline \n")
        source.link(caps_v4l2src)
        caps_v4l2src.link(vidconvsrc)
        vidconvsrc.link(nvvidconvsrc)
        nvvidconvsrc.link(caps_vidconvsrc)

        sinkpad = streammux.get_request_pad("sink_0")
        if not sinkpad:
            sys.stderr.write(" Unable to get the sink pad of streammux \n")
        
        srcpad = caps_vidconvsrc.get_static_pad("src")
        if not srcpad:
            sys.stderr.write(" Unable to get source pad of decoder \n")
        srcpad.link(sinkpad)
        streammux.link(tee)
        tee.link(queue1)
        tee.link(queue2)
        queue1.link(pgie)
        queue2.link(pgie2)
        pgie.link(tracker)
        tracker.link(sgie1)
        #sgie1.link(sgie2)
        #sgie2.link(sgie3)
        #sgie3.link(nvvidconv1)
        sgie1.link(nvvidconv1)
        nvvidconv1.link(nvosd1)

        pgie2.link(nvvidconv2)
        nvvidconv2.link(nvosd2)

        if is_aarch64():
            nvosd1.link(transform1)
            transform1.link(sink1)
            nvosd2.link(transform2)
            transform2.link(sink2)
        else:
            nvosd1.link(sink1)
            nvosd2.link(sink2)


        # create and event loop and feed gstreamer bus mesages to it
        self.loop = GObject.MainLoop()
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect ("message", bus_call, self.loop)

        # Lets add probe to get informed of the meta data generated, we add probe to
        # the sink pad of the osd element, since by that time, the buffer would have
        # had got all the metadata.
        osdsinkpad1 = nvosd1.get_static_pad("sink")
        if not osdsinkpad1:
            sys.stderr.write(" Unable to get sink pad of nvosd \n")
        osdsinkpad1.add_probe(Gst.PadProbeType.BUFFER, self.osd_sink_pad_buffer_probe, 0)

        osdsinkpad2 = nvosd2.get_static_pad("sink")
        if not osdsinkpad2:
            sys.stderr.write(" Unable to get sink pad of nvosd \n")
        osdsinkpad2.add_probe(Gst.PadProbeType.BUFFER, self.osd_sink_pad_buffer_probe, 0)


    def start_pipeline(self):
        print("Starting pipeline \n")
        # start play back and listen to events
        self.pipeline.set_state(Gst.State.PLAYING)
        try:
            self.loop.run()
        except:
            pass
        # cleanup
        self.pipeline.set_state(Gst.State.NULL)

