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

# This node performs detection and classification inference on multiple input video files and publishes results to topics multi_detection and multi_classification respectively

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

from ctypes import *
import time
import math
import numpy as np
import cv2
import os
fps_streams = {}
frame_count = {}
saved_count = {}

MAX_DISPLAY_LEN=64
MUXER_OUTPUT_WIDTH=1920
MUXER_OUTPUT_HEIGHT=1080
MUXER_BATCH_TIMEOUT_USEC=4000000
TILED_OUTPUT_WIDTH=1920
TILED_OUTPUT_HEIGHT=1080
GST_CAPS_FEATURES_NVMM="memory:NVMM"
pgie_classes_str= ["Vehicle", "TwoWheeler", "Person","RoadSign"]


PGIE_CLASS_ID_VEHICLE = 0
PGIE_CLASS_ID_BICYCLE = 1
PGIE_CLASS_ID_PERSON = 2
PGIE_CLASS_ID_ROADSIGN = 3

location = os.getcwd() + "/src/ros2_deepstream/config_files/"
class_obj = (open(location+'object_labels.txt').readline().rstrip('\n')).split(';')

class_color = (open(location+'color_labels.txt').readline().rstrip('\n')).split(';')

class_make = (open(location+'make_labels.txt').readline().rstrip('\n')).split(';')

class_type = (open(location+'type_labels.txt').readline().rstrip('\n')).split(';')


class InferencePublisher(Node):
    # tiler_sink_pad_buffer_probe  will extract metadata received on tiler src pad
    # and update params for drawing rectangle, object information etc.
    def tiler_sink_pad_buffer_probe(self,pad,info,u_data):
        frame_number=0
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
            l_obj=frame_meta.obj_meta_list
            num_rects = frame_meta.num_obj_meta
            is_first_obj = True
            save_image = False
            obj_counter = {
            PGIE_CLASS_ID_VEHICLE:0,
            PGIE_CLASS_ID_BICYCLE:0,
            PGIE_CLASS_ID_PERSON:0,
            PGIE_CLASS_ID_ROADSIGN:0
            }


            # Message for output of detection inference
            msg = Detection2DArray()
            while l_obj is not None:
                try:
                    # Casting l_obj.data to pyds.NvDsObjectMeta
                    obj_meta=pyds.NvDsObjectMeta.cast(l_obj.data)
                    l_classifier = obj_meta.classifier_meta_list
                    # If object is a car (class ID 0), perform attribute classification
                    if obj_meta.class_id == 0 and l_classifier is not None:
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
                            
                            if classifier_id == 2:
                                result.id = class_color[classifier_class]
                            elif classifier_id == 3:
                                result.id = class_make[classifier_class]
                            else:
                                result.id = class_type[classifier_class]
                            
                            result.score = label_info.result_prob
                            msg2.results.append(result)
                            l_classifier = l_classifier.next
                        
                        self.publisher_classification.publish(msg2)

                except StopIteration:
                    break

                obj_counter[obj_meta.class_id] += 1

                # Creating message for output of detection inference
                result = ObjectHypothesisWithPose()
                result.id = str(class_obj[obj_meta.class_id])
                result.score = obj_meta.confidence
                
                left = obj_meta.rect_params.left
                top = obj_meta.rect_params.top
                width = obj_meta.rect_params.width
                height = obj_meta.rect_params.height
                bounding_box = BoundingBox2D()
                bounding_box.center.x = float(left + (width/2)) 
                bounding_box.center.y = float(top - (height/2))
                bounding_box.size_x = width
                bounding_box.size_y = height
                
                detection = Detection2D()
                detection.results.append(result)
                detection.bbox = bounding_box
                msg.detections.append(detection)


                # Periodically check for objects with borderline confidence value that may be false positive detections.
                # If such detections are found, annotate the frame with bboxes and confidence value.
                # Save the annotated frame to file.
                if((saved_count["stream_"+str(frame_meta.pad_index)]%30==0) and (obj_meta.confidence>0.3 and obj_meta.confidence<0.31)):
                    if is_first_obj:
                        is_first_obj = False
                        # Getting Image data using nvbufsurface
                        # the input should be address of buffer and batch_id
                        n_frame=pyds.get_nvds_buf_surface(hash(gst_buffer),frame_meta.batch_id)
                        #convert python array into numy array format.
                        frame_image=np.array(n_frame,copy=True,order='C')
                        #covert the array into cv2 default color format
                        frame_image=cv2.cvtColor(frame_image,cv2.COLOR_RGBA2BGRA)

                    save_image = True
                    frame_image=draw_bounding_boxes(frame_image,obj_meta,obj_meta.confidence)
                try:
                    l_obj=l_obj.next
                except StopIteration:
                    break


            # Get frame rate through this probe
            fps_streams["stream{0}".format(frame_meta.pad_index)].get_fps()

            # Publishing message with output of detection inference        
            self.publisher_detection.publish(msg)


            if save_image:
                cv2.imwrite(folder_name+"/stream_"+str(frame_meta.pad_index)+"/frame_"+str(frame_number)+".jpg",frame_image)
            saved_count["stream_"+str(frame_meta.pad_index)]+=1
            try:
                l_frame=l_frame.next
            except StopIteration:
                break

        return Gst.PadProbeReturn.OK

    def draw_bounding_boxes(image,obj_meta,confidence):
        confidence='{0:.2f}'.format(confidence)
        rect_params=obj_meta.rect_params
        top=int(rect_params.top)
        left=int(rect_params.left)
        width=int(rect_params.width)
        height=int(rect_params.height)
        obj_name=pgie_classes_str[obj_meta.class_id]
        image=cv2.rectangle(image,(left,top),(left+width,top+height),(0,0,255,0),2)
        # Note that on some systems cv2.putText erroneously draws horizontal lines across the image
        image=cv2.putText(image,obj_name+',C='+str(confidence),(left-10,top-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255,0),2)
        return image

    def cb_newpad(self, decodebin, decoder_src_pad,data):
        print("In cb_newpad\n")
        caps=decoder_src_pad.get_current_caps()
        gststruct=caps.get_structure(0)
        gstname=gststruct.get_name()
        source_bin=data
        features=caps.get_features(0)

        # Need to check if the pad created by the decodebin is for video and not
        # audio.
        if(gstname.find("video")!=-1):
            # Link the decodebin pad only if decodebin has picked nvidia
            # decoder plugin nvdec_*. We do this by checking if the pad caps contain
            # NVMM memory features.
            if features.contains("memory:NVMM"):
                # Get the source bin ghost pad
                bin_ghost_pad=source_bin.get_static_pad("src")
                if not bin_ghost_pad.set_target(decoder_src_pad):
                    sys.stderr.write("Failed to link decoder src pad to source bin ghost pad\n")
            else:
                sys.stderr.write(" Error: Decodebin did not pick nvidia decoder plugin.\n")

    def decodebin_child_added(self,child_proxy,Object,name,user_data):
        print("Decodebin child added:", name, "\n")
        if(name.find("decodebin") != -1):
            Object.connect("child-added",decodebin_child_added,user_data)
        if(is_aarch64() and name.find("nvv4l2decoder") != -1):
            print("Seting bufapi_version\n")
            Object.set_property("bufapi-version",True)

    def create_source_bin(self,index,uri):
        print("Creating source bin")

        # Create a source GstBin to abstract this bin's content from the rest of the
        # pipeline
        bin_name="source-bin-%02d" %index
        print(bin_name)
        nbin=Gst.Bin.new(bin_name)
        if not nbin:
            sys.stderr.write(" Unable to create source bin \n")

        # Source element for reading from the uri.
        # We will use decodebin and let it figure out the container format of the
        # stream and the codec and plug the appropriate demux and decode plugins.
        uri_decode_bin=Gst.ElementFactory.make("uridecodebin", "uri-decode-bin")
        if not uri_decode_bin:
            sys.stderr.write(" Unable to create uri decode bin \n")
        # We set the input uri to the source element
        uri_decode_bin.set_property("uri",uri)
        # Connect to the "pad-added" signal of the decodebin which generates a
        # callback once a new pad for raw data has beed created by the decodebin
        uri_decode_bin.connect("pad-added",self.cb_newpad,nbin)
        uri_decode_bin.connect("child-added",self.decodebin_child_added,nbin)

        # We need to create a ghost pad for the source bin which will act as a proxy
        # for the video decoder src pad. The ghost pad will not have a target right
        # now. Once the decode bin creates the video decoder and generates the
        # cb_newpad callback, we will set the ghost pad target to the video decoder
        # src pad.
        Gst.Bin.add(nbin,uri_decode_bin)
        bin_pad=nbin.add_pad(Gst.GhostPad.new_no_target("src",Gst.PadDirection.SRC))
        if not bin_pad:
            sys.stderr.write(" Failed to add ghost pad in source bin \n")
            return None
        return nbin 


    def __init__(self):
        super().__init__('inference_publisher')

        self.declare_parameter('input_sources')
        input_sources = self.get_parameter('input_sources').value
        number_sources = len(input_sources)

        for i in range(number_sources):
            fps_streams["stream{0}".format(i)]=GETFPS(i)


        self.publisher_detection = self.create_publisher(Detection2DArray, 'multi_detection', 10)

        self.publisher_classification = self.create_publisher(Classification2D, 'multi_classification', 10)
    
        # Standard GStreamer initialization
        GObject.threads_init()
        Gst.init(None)

        # Create gstreamer elements
        # Create Pipeline element that will form a connection of other elements
        print("Creating Pipeline \n ")
        self.pipeline = Gst.Pipeline()
        if not self.pipeline:
            sys.stderr.write(" Unable to create Pipeline \n")

        print("Creating streamux \n ")
        # Create nvstreammux instance to form batches from one or more sources.
        streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
        if not streammux:
            sys.stderr.write(" Unable to create NvStreamMux \n")
        self.pipeline.add(streammux)
        

        for i in range(number_sources):
            frame_count["stream_"+str(i)]=0
            saved_count["stream_"+str(i)]=0
            print("Creating source_bin ",i," \n ")
            uri_name=input_sources[i]
            if uri_name.find("rtsp://") == 0 :
                is_live = True
            source_bin=self.create_source_bin(i, uri_name)
            if not source_bin:
                sys.stderr.write("Unable to create source bin \n")
            self.pipeline.add(source_bin)
            padname="sink_%u" %i
            sinkpad= streammux.get_request_pad(padname)
            if not sinkpad:
                sys.stderr.write("Unable to create sink pad bin \n")
            srcpad=source_bin.get_static_pad("src")
            if not srcpad:
                sys.stderr.write("Unable to create src pad bin \n")
            srcpad.link(sinkpad)


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

        sgie2 = Gst.ElementFactory.make("nvinfer", "secondary2-nvinference-engine")
        if not sgie1:
            sys.stderr.write(" Unable to make sgie2 \n")

        sgie3 = Gst.ElementFactory.make("nvinfer", "secondary3-nvinference-engine")
        if not sgie3:
            sys.stderr.write(" Unable to make sgie3 \n")

        pgie2 = Gst.ElementFactory.make("nvinfer", "primary-inference2")
        if not pgie2:
            sys.stderr.write(" Unable to create pgie2 \n")

        nvvidconv1 = Gst.ElementFactory.make("nvvideoconvert", "convertor1")
        if not nvvidconv1:
            sys.stderr.write(" Unable to create nvvidconv1 \n")

        print("Creating filter1 \n ")
        caps1 = Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA")
        filter1 = Gst.ElementFactory.make("capsfilter", "filter1")
        if not filter1:
            sys.stderr.write(" Unable to get the caps filter1 \n")
        filter1.set_property("caps", caps1)

        print("Creating tiler1 \n ")
        tiler1=Gst.ElementFactory.make("nvmultistreamtiler", "nvtiler1")
        if not tiler1:
            sys.stderr.write(" Unable to create tiler1 \n")
        
        print("Creating nvvidconv_1 \n ")
        nvvidconv_1 = Gst.ElementFactory.make("nvvideoconvert", "convertor_1")
        if not nvvidconv_1:
            sys.stderr.write(" Unable to create nvvidconv_1 \n")

        nvvidconv2 = Gst.ElementFactory.make("nvvideoconvert", "convertor2")
        if not nvvidconv2:
            sys.stderr.write(" Unable to create nvvidconv2 \n")

        caps2 = Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA")
        filter2 = Gst.ElementFactory.make("capsfilter", "filter2")
        if not filter2:
            sys.stderr.write(" Unable to get the caps filter2 \n")
        filter2.set_property("caps", caps2)
        
        print("Creating tiler2 \n ")
        tiler2=Gst.ElementFactory.make("nvmultistreamtiler", "nvtiler2")
        if not tiler2:
            sys.stderr.write(" Unable to create tiler2 \n")
        
        print("Creating nvvidconv_2 \n ")
        nvvidconv_2 = Gst.ElementFactory.make("nvvideoconvert", "convertor_2")
        if not nvvidconv_2:
            sys.stderr.write(" Unable to create nvvidconv_2 \n")



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

        streammux.set_property('width', 1920)
        streammux.set_property('height', 1080)
        streammux.set_property('batch-size', number_sources)
        streammux.set_property('batched-push-timeout', 4000000)

        #Set properties of pgie and sgie
        location = os.getcwd() + "/src/ros2_deepstream/config_files/"
        pgie.set_property('config-file-path', location+"dstest2_pgie_config.txt")
        pgie_batch_size=pgie.get_property("batch-size")
        if(pgie_batch_size != number_sources):
            print("WARNING: Overriding infer-config batch-size",pgie_batch_size," with number of sources ", number_sources," \n")
            pgie.set_property("batch-size",number_sources)

        sgie1.set_property('config-file-path', location+"dstest2_sgie1_config.txt")
        sgie2.set_property('config-file-path', location+"dstest2_sgie2_config.txt")
        sgie3.set_property('config-file-path', location+"dstest2_sgie3_config.txt")
        pgie2.set_property('config-file-path', location+"dstest1_pgie_config.txt")
        sink1.set_property('sync', False)
        sink2.set_property('sync', False)


        tiler_rows=int(math.sqrt(number_sources))
        tiler_columns=int(math.ceil((1.0*number_sources)/tiler_rows))
        tiler1.set_property("rows",tiler_rows)
        tiler1.set_property("columns",tiler_columns)
        tiler1.set_property("width", TILED_OUTPUT_WIDTH)
        tiler1.set_property("height", TILED_OUTPUT_HEIGHT)

        tiler2.set_property("rows",tiler_rows)
        tiler2.set_property("columns",tiler_columns)
        tiler2.set_property("width", TILED_OUTPUT_WIDTH)
        tiler2.set_property("height", TILED_OUTPUT_HEIGHT)

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
        self.pipeline.add(pgie)
        self.pipeline.add(pgie2)
        self.pipeline.add(tracker)
        self.pipeline.add(sgie1)
        self.pipeline.add(sgie2)
        self.pipeline.add(sgie3)
        self.pipeline.add(nvvidconv1)
        self.pipeline.add(nvvidconv2)
        self.pipeline.add(nvosd1)
        self.pipeline.add(nvosd2)
        self.pipeline.add(sink1)
        self.pipeline.add(sink2)
        self.pipeline.add(tee)
        self.pipeline.add(queue1)
        self.pipeline.add(queue2)
        self.pipeline.add(tiler1)
        self.pipeline.add(tiler2)
        self.pipeline.add(filter1)
        self.pipeline.add(filter2)
        self.pipeline.add(nvvidconv_1)
        self.pipeline.add(nvvidconv_2)


        if is_aarch64():
            self.pipeline.add(transform1)
            self.pipeline.add(transform2)

        # Link the elements together
        print("Linking elements in the Pipeline \n")
        streammux.link(tee)
        tee.link(queue1)
        tee.link(queue2)
        queue1.link(pgie)
        queue2.link(pgie2)
        pgie.link(tracker)
        tracker.link(sgie1)
        sgie1.link(sgie2)
        sgie2.link(sgie3)
        sgie3.link(nvvidconv1)
        nvvidconv1.link(filter1)
        filter1.link(tiler1)
        tiler1.link(nvvidconv_1)
        nvvidconv_1.link(nvosd1)

        pgie2.link(nvvidconv2)
        nvvidconv2.link(filter2)
        filter2.link(tiler2)
        tiler2.link(nvvidconv_2)
        nvvidconv_2.link(nvosd2)


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

        tiler_sink_pad_1=tiler1.get_static_pad("sink")
        if not tiler_sink_pad_1:
            sys.stderr.write(" Unable to get src pad \n")
        else:
            tiler_sink_pad_1.add_probe(Gst.PadProbeType.BUFFER, self.tiler_sink_pad_buffer_probe, 0)

        tiler_sink_pad_2=tiler2.get_static_pad("sink")
        if not tiler_sink_pad_2:
            sys.stderr.write(" Unable to get src pad \n")
        else:
           tiler_sink_pad_2.add_probe(Gst.PadProbeType.BUFFER, self.tiler_sink_pad_buffer_probe, 0)


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
