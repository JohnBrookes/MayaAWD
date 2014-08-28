import re
import sys
import math
import os.path
from time import time

import maya.OpenMaya as om
import maya.cmds as mc
import maya.OpenMayaAnim as omanim
import maya.OpenMayaMPx as OpenMayaMPx
import pymel.core.datatypes

import pyawd
from pyawd.core import *
from pyawd.anim import *
from pyawd.scene import *
from pyawd.geom import *
from pyawd.material import *
from pyawd.utils.math import *



b_start = 0.0

def benchmark_start():
    global b_start
    b_start = time()

def benchmark_stop():
    global b_start

    dur = time() - b_start
    b_start = 0.0
    return dur

def benchmark_print():
    dur = benchmark_stop()
    print('Duration: %fs' % dur)


class MayaAWDFileTranslator(OpenMayaMPx.MPxFileTranslator):
    def writer(self, file, options, mode):
        file_path = file.resolvedFullName()
        base_path = os.path.dirname(file_path)

        opts = self.parse_opts(options)

        def o(key, defval=None):
            'Get option or default value'
            if key in opts:
                return opts[key]
            else:
                return defval


        with open(file_path, 'wb') as file:
            comp_str = o('compression', 'none')
            compression = UNCOMPRESSED
            if comp_str == 'deflate':
                compression = DEFLATE
            elif comp_str == 'lzma':
                compression = LZMA

            wide_mtx = int(o('wide_mtx', False))
            wide_geom = int(o('wide_geom', False))
                
            exporter = MayaAWDExporter(file, compression, wide_geom=wide_geom, wide_mtx=wide_mtx)
            exporter.include_geom = bool(o('inc_geom', False))
            exporter.include_scene = bool(o('inc_scene', False))
            exporter.flatten_untransformed = bool(o('flatten_untransformed', False))
            exporter.replace_exrefs = bool(o('replace_exrefs', False))
            exporter.include_uvanim = bool(o('inc_uvanim', False))
            exporter.include_skelanim = bool(o('inc_skelanim', False))
            exporter.include_skeletons = bool(o('inc_skeletons', False))
            exporter.include_materials = bool(o('inc_materials', False))
            exporter.include_cameras = bool(o('inc_cams', False))
            exporter.include_lights = bool(o('inc_lights', False))
            exporter.embed_textures = bool(o('embed_textures', False))
            exporter.alpha_blending = bool(o('alpha_blending', False))
            exporter.alpha_threshold = float(o('alpha_threshold', 0.0))
            exporter.include_attr = bool(o('inc_attr', False))

            if exporter.include_attr:
                exporter.user_ns = AWDNamespace(str(o('attrns', '')))

            if exporter.include_skelanim:
                exporter.animation_sequences = self.read_sequences(o('seqsrc'), base_path)
                exporter.joints_per_vert = int(o('jointspervert', 3))

            exporter.export(None)

            #TODO: Check whether to copy viewer
            if False:
                import shutil
                import subprocess

                pyawd_path = pyawd.__path__[0]
                viewer_path = os.path.normpath(os.path.join(pyawd_path, '..', 'mayaawd'))
                out_path = os.path.dirname(file_path)
                out_name = os.path.basename(os.path.splitext(file_path)[0])

                #TODO: Check local/network
                viewer_name = 'viewer_l.swf'

                shutil.copyfile(os.path.join(viewer_path, viewer_name), os.path.join(out_path, 'viewer.swf'))
                shutil.copyfile(os.path.join(viewer_path, 'swfobject.js'), os.path.join(out_path, 'swfobject.js'))

                html_template = os.path.join(viewer_path, 'template.html')
                html_output = os.path.splitext(file_path)[0] + '.html'

                # TODO: Fetch color from options
                bg_color = '000000'

                with open(html_template, 'r') as html_in:
                    with open(html_output, 'w') as html_out:
                        for line in html_in:
                            line = line.replace('%NAME%', out_name)
                            line = line.replace('%COLOR%', bg_color) 
                            html_out.write(line)
                
                try:
                    # Windows?
                    os.startfile(html_output)
                except AttributeError:
                    # Mac OS X
                    subprocess.call(['open', html_output])


    def defaultExtension(self):
        return 'awd'

    def haveWriteMethod(self):
        return True

    def parse_opts(self, opt_str):
        if opt_str[0]==';':
            opt_str=opt_str[1:]

        fields = re.split('(?<!\\\)&', str(opt_str))
        return dict([ re.split('(?<!\\\)=', pair) for pair in fields ])

    def read_sequences(self, seq_path, base_path):
        sequences = []
        if seq_path is not None:
            if not os.path.isabs(seq_path):
                # Look for this file in a list of different locations,
                # and use the first one in which it exists.
                existed = False
                bases = [
                    mc.workspace(q=True, rd=True),
                    os.path.join(mc.workspace(q=True, rd=True), mc.workspace('mayaAscii', q=True, fre=True)),
                    os.path.join(mc.workspace(q=True, rd=True), mc.workspace('AWD2', q=True, fre=True)),
                    base_path
                ]

                for base in bases:
                    new_path = os.path.join(base, seq_path)
                    print('Looking for sequence file in %s' % new_path)
                    if os.path.exists(new_path) and os.path.isfile(new_path):
                        existed = True
                        seq_path = new_path
                        break

                if not existed:
                    mc.warning('Could not find sequence file "%s. Will not export animation."' % seq_path)
                    return []

            try:
                with open(seq_path, 'r') as seqf:
                    lines = seqf.readlines()
                    for line in lines:
                        # Skip comments
                        if line[0] == '#':
                            continue

                        line_fields = re.split('[^a-zA-Z0-9]', line.strip())
                        sequences.append((line_fields[0], int(line_fields[1]), int(line_fields[2])))
            except:
                pass

        return sequences

def ftCreator():
    return OpenMayaMPx.asMPxPtr( MayaAWDFileTranslator() )

def initializePlugin(mobject):
    mplugin = OpenMayaMPx.MFnPlugin(mobject, 'Away3D', '1.0')
    stat = mplugin.registerFileTranslator('AWD2', 'none', ftCreator, 'MayaAWDExporterUI')

    return stat
    
def uninitializePlugin(mobject):
    mplugin = OpenMayaMPx.MFnPlugin(mobject)
    stat = mplugin.deregisterFileTranslator('AWD2')

    return stat



class MayaAWDBlockCache:
    '''A cache of already created AWD blocks, and their connection to
        nodes in the Maya DAG. The cache should always be checked before
        creating a blocks, so that blocks can be reused within the file
        when possible.'''
    
    def __init__(self):
        self.__cache = []

    def get(self, path):
        block = None
        for item in self.__cache:
            if item[0] == path:
                block = item[1]
                break

        return block
            

    def add(self, path, block):
        if self.get(path) is None:
            self.__cache.append((path, block))
        


class MayaAWDExporter:
    def __init__(self, file, compression, wide_geom=False, wide_mtx=False):
        self.file = file
        self.block_cache = MayaAWDBlockCache()
        self.skeleton_paths = []
        self.joint_indices = {}
        self.mesh_vert_indices = {}

        self.include_attr = False
        self.include_geom = False
        self.include_scene = False
        self.flatten_untransformed = False
        self.replace_exrefs = False
        self.include_uvanim = False
        self.include_skelanim = False
        self.include_skeletons = False
        self.include_materials = False
        self.include_cameras = False
        self.include_lights = False
        self.embed_textures = False
        self.animation_sequences = []

        self.has_skelanim = False

        self.awd = AWD(compression=compression, wide_geom=wide_geom, wide_mtx=wide_mtx)


    def export(self, selection):
        # Assume that bind pose is on frame 1
        om.MGlobal.viewFrame(0)
  
        self.export_scene()

        if self.include_skeletons:
            self.export_skeletons()

        if self.include_skelanim and self.has_skelanim:
            self.export_animation(self.animation_sequences)
 
        self.awd.flush(self.file)

    def export_scene(self):
        dag_it = om.MItDag(om.MItDag.kDepthFirst)
        while not dag_it.isDone():
            visible = False
            try:
                attr0 = '%s.visibility' % dag_it.partialPathName()
                attr1 = '%s.ovv' % dag_it.partialPathName()
                visible = mc.getAttr(attr0) and mc.getAttr(attr1)
            except:
                pass

            if visible:
                if dag_it.currentItem().hasFn(om.MFn.kTransform):
                    transform = dag_it.fullPathName()

                    print('')
                    print('================================================')
                    print('export %s' % dag_it.fullPathName())
                    print('================================================')

                    def find_nearest_cached_ancestor(child_dag_fn):
                        if child_dag_fn.parentCount() > 0:
                            parent_dag_fn = om.MFnDagNode(child_dag_fn.parent(0))
                            print('looking in cache for "%s"' % parent_dag_fn.fullPathName())
                            awd_parent = self.block_cache.get(parent_dag_fn.fullPathName())
                            if awd_parent is not None:
                                return awd_parent
                            else:
                                return find_nearest_cached_ancestor(parent_dag_fn)
                        else:
                            return None
                        

                    dag_fn = om.MFnDagNode(dag_it.currentItem())
                    awd_parent = find_nearest_cached_ancestor(dag_fn)
                    shapes = mc.listRelatives(transform, s=True, f=True)
                    if shapes is not None:
                        shape = shapes[0]
                        api_type = mc.nodeType(shape, api=True)
                        if api_type == 'kMesh':
                            self.export_mesh(transform, shape, awd_parent)
                        elif api_type == 'kCamera' and self.include_cameras:
                            # Cameras for some reason are "shapes" in Maya
                            self.export_camera(transform, awd_parent)
                    elif not dag_it.currentItem().hasFn(om.MFn.kJoint):
                        # Container!
                        mtx = mc.xform(transform, q=True, m=True)

                        #Skip this container if untransformed and transformation is identity
                        id_mtx = [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]
                        if not (self.flatten_untransformed and mtx == id_mtx):
                            ctr = AWDContainer(name=dag_it.partialPathName(), transform=self.mtx_list2awd(mtx))
                            print('saving in cache with id %s' % transform)
                            self.block_cache.add(transform, ctr)
                            if awd_parent is not None:
                                awd_parent.add_child(ctr)
                            else:
                                self.awd.add_scene_block(ctr)

                            self.set_attributes(transform, ctr)
  
            else:
                if dag_it.fullPathName(): # Not root
                    # Stop iterating this branch of the tree
                    dag_it.prune()
                print('skipping invisible %s' % dag_it.fullPathName())

            dag_it.next()


    def export_camera(self, transform, awd_parent):
        mtx = mc.xform(transform, q=True, m=True)
        cam = AWDCamera(self.get_name(transform), AWDMatrix3x4(mtx))
        cam.type = CAM_FREE
        cam.lens = LENS_PERSPECTIVE
        cam.fov = mc.camera(transform, q=True, vfv=True)

        if awd_parent is not None:
            awd_parent.add_child(cam)
        else:
            self.awd.add_scene_block(cam)
        

    def export_skeletons(self):
        dag_it = om.MItDependencyNodes(om.MFn.kSkinClusterFilter)
        while not dag_it.isDone():
            obj = dag_it.thisNode()
            joints = om.MDagPathArray()
 
            skin_fn = omanim.MFnSkinCluster(obj)
            num_joints = skin_fn.influenceObjects(joints)
 
 
            # Loop through joints and look in block cache whether
            # a skeleton for this joint has been exported. If not,
            # we will ignore this binding altogether.
            skel = None
            #print('found skin cluster for %s!' % skel)
            for i in range(num_joints):
                #print('affected joint: %s' % joints[i].fullPathName())
                skel = self.block_cache.get(self.get_skeleton_root(joints[i].fullPathName()))
                if skel is not None:
                    break
 
            # Skeleton was found
            if skel is not None:
                #print('found skeleton in cache!')
                #print('num joints: %d' % num_joints)
 
                # Loop through meshes that are influenced by this
                # skeleton, and add weight stream to their mesh data
                num_geoms = skin_fn.numOutputConnections()
                #print('num geoms: %d' % num_geoms)
                for i in range(num_geoms):
                    skin_path = om.MDagPath()
                    skin_fn.getPathAtIndex(i, skin_path)
                    vert_it = om.MItMeshVertex(skin_path)
 
                    #print('skin obj: %s' % skin_path.fullPathName())
 
                    # Check whether a mesh data for this geometry has
                    # been added to the block cache. If not, bindings
                    # for this mesh can be ignored.
                    md = self.block_cache.get(self.get_name(skin_path.fullPathName()))
                    if md is not None:
                        #print('found mesh in cache!')
                        weight_data = []
                        index_data = []

                        self.has_skelanim = True
 
                        while not vert_it.isDone():
                            comp = vert_it.currentItem()
                            weights = om.MDoubleArray()
                            weight_objs = []
 
                            #script_util = om.MScriptUtil()
                            for ii in range(num_joints):
                                skin_fn.getWeights(skin_path, comp, ii, weights)
                                joint_name = joints[ii].fullPathName()
                               # print('JB %s' % joint_name.split('|')[-1])
                                joint_idx = self.joint_indices[joint_name.split('|')[-1]]
                                #print('ok here %s' % joint_idx)
                                weight_objs.append( (joint_idx, weights[0]) )
 
                            def comp_weight_objs(wo0, wo1):
                                if wo0[1] > wo1[1]: return -1
                                else: return 1
 
                            weight_objs.sort(comp_weight_objs)
 
                            # Normalize top weights
                            weight_objs = weight_objs[0:self.joints_per_vert]
                            sum_obj = reduce(lambda w0,w1: (0, w0[1]+w1[1]), weight_objs)
                            if sum_obj[1] > 0.0:
                                weight_objs = map(lambda w: (w[0], w[1] / sum_obj[1]), weight_objs)
 
                            # Add more empty weight objects if too few
                            if len(weight_objs) != self.joints_per_vert:
                                weight_objs.extend([(0,0)] * (self.joints_per_vert - len(weight_objs)))
 
                            for w_obj in weight_objs:
                                index_data.append(w_obj[0])
                                weight_data.append(w_obj[1])
 
                            vert_it.next()
 
                        weight_stream = []
                        index_stream = []
 
                        # This list contains the old-index of each vertex in the AWD vertex stream
                        vert_indices = self.mesh_vert_indices[skin_path.fullPathName()]
                        for idx in vert_indices:
                            start_idx = idx*self.joints_per_vert
                            end_idx = start_idx + self.joints_per_vert
                            w_tuple = weight_data[start_idx:end_idx]
                            i_tuple = index_data[start_idx:end_idx]
                            weight_stream.extend(w_tuple)
                            index_stream.extend(i_tuple)
 
                        if len(md) == 1:
                            print('Setting streams!')
                            sub = md[0]
                            sub.add_stream(pyawd.geom.STR_JOINT_WEIGHTS, weight_stream)
                            sub.add_stream(pyawd.geom.STR_JOINT_INDICES, index_stream)
                        else:
                            print('skinning not implemented for meshes with <> 1 sub-mesh')
 
            dag_it.next()
        


    def export_animation(self, sequences):
        #TODO: Don't hard-code these.
        #animated_materials = [ 'MAT_BlueEye_L', 'MAT_BlueEye_R' ]
        #animated_materials = [ 'MAT_BrownEye_L', 'MAT_BrownEye_R' ]
        animated_materials = []
 
        for seq in sequences:
            frame_idx = seq[1]
            end_frame = seq[2]
 
            print('exporting sequence "%s" (%d-%d)' % seq)
 
            if len(self.skeleton_paths) > 0:
                anim = AWDSkeletonAnimation(seq[0])
                self.awd.add_skeleton_anim(anim)
 
            uvanims = []
            for mat in animated_materials:
                uvanim = AWDUVAnimation(mat.replace('MAT', 'UVANIM')+'_'+seq[0])
                uvanims.append(uvanim)
                self.awd.add_uv_anim(uvanim)
 
            while frame_idx <= end_frame:
                om.MGlobal.viewFrame(frame_idx)
 
                self.sample_materials(animated_materials, uvanims)
 
                for skeleton_path in self.skeleton_paths:
                    def get_all_transforms(joint_path, list):
                        mtx_list = mc.xform(joint_path, q=True, m=True)
                        list.append( self.mtx_list2awd(mtx_list))
 
                        children = mc.listRelatives(joint_path, type='joint')
                        if children is not None:
                            for child in children:
                                get_all_transforms(child, list)
 
                    skel_pose = AWDSkeletonPose()
 
                    all_transforms = []
                    get_all_transforms(skeleton_path, all_transforms)
                    for tf in all_transforms:
                        skel_pose.add_joint_transform(tf)
 
                    #TODO: Don't hard-code duration
                    anim.add_frame(skel_pose, 40)
                    self.awd.add_skeleton_pose(skel_pose)
 
                # Move to next frame
                frame_idx += 1
 
        

    def export_mesh(self, transform, shape, awd_ctr):
        try:
            mtx = mc.xform(transform, q=True, m=True)
        except:
            print('skipping invalid %s' % transform)
 
        tf_name = self.get_name(transform)
        sh_name = self.get_name(shape)

        tf_is_ref = mc.referenceQuery(transform, inr=True)
        sh_is_ref = mc.referenceQuery(shape, inr=True)
        if (tf_is_ref or sh_is_ref) and self.replace_exrefs:
            # This is an external reference, and it should be
            # replaced with an empty container in the AWD file
            ctr = AWDContainer(name=tf_name, transform=AWDMatrix3x4(mtx))
            self.set_attributes(transform, ctr)
            self.block_cache.add(transform, ctr)
            if awd_ctr is not None:
                awd_ctr.add_child(ctr)
            else:
                self.awd.add_scene_block(ctr)

        else:
            md = self.block_cache.get(sh_name)
            if md is None:
                print('Creating mesh data %s' % sh_name)
                md = AWDTriGeom(sh_name)
                md.bind_matrix = AWDMatrix3x4(mtx)
                self.export_mesh_data(md, shape)
                self.awd.add_tri_geom(md)
                self.block_cache.add(sh_name, md)
 
            inst = AWDMeshInst(md, tf_name, self.mtx_list2awd(mtx))
 
            self.set_attributes(transform, inst)

            # Look for materials
            if self.include_materials:
                self.export_materials(transform, inst)
 
            self.block_cache.add(transform, inst)
            if awd_ctr is not None:
                awd_ctr.add_child(inst)
            else:
                self.awd.add_scene_block(inst)
 
            if self.include_skeletons:
                history = mc.listHistory(transform)
                clusters = mc.ls(history, type='skinCluster')
                if len(clusters) > 0:
                    #TODO: Deal with multiple clusters?
                    sc = clusters[0]
 
                    influences = mc.skinCluster(sc, q=True, inf=True)
                    if len(influences) > 0:
                        skel_path = self.get_skeleton_root(influences[0])
 
                        if self.block_cache.get(skel_path) is None:
                            self.export_skeleton(skel_path)
 
    def export_materials(self, transform, awd_inst):
        sets = mc.listSets(object=transform, t=1, ets=True)
        if sets is not None:
            for set in sets:
                if mc.nodeType(set)=='shadingEngine':
                    tex = None
                    mat = None
                    mat_his = mc.listHistory(set)
                    for state in mat_his:
                        state_type = mc.nodeType(state)
 
                        if state_type == 'lambert' or state_type == 'blinn' or state_type == 'phong':
                            mat = self.block_cache.get(state)
                            if mat is None:
                                mat = AWDMaterial(AWDMaterial.BITMAP, name=self.get_name(state))
                                self.awd.add_material(mat)
                                self.block_cache.add(state, mat)
                                print('created material')

                                if self.alpha_blending or self.alpha_threshold > 0.0:
                                    # Check if transparency is an input (rather than scalars)
                                    # in which case the material needs to be marked as transparent,
                                    # to indicate that the texture's alpha channel should be used.
                                    tr_input = mc.connectionInfo('%s.it' % state, isDestination=True)
                                    if tr_input:
                                        if self.alpha_threshold > 0.0:
                                            mat.alpha_threshold = self.alpha_threshold
                                        else:
                                            mat.alpha_blending = True
 
                            awd_inst.materials.append(mat)
                            print('adding material ' + state)
                            
                        # Only check the first file, which will likely be the color input.
                        # TODO: This needs to be solved in a prettier way for normal maps
                        # and other inputs like that.
                        elif state_type == 'file' and tex is None:
                            tex = self.block_cache.get(state)
                            if tex is None:
                                tex_abs_path = str(mc.getAttr(state+'.fileTextureName'))
                                if self.embed_textures:
                                    tex = AWDBitmapTexture(AWDBitmapTexture.EMBED, name=self.get_name(state))
                                    tex.embed_file(tex_abs_path)
                                    print('embedding %s' % tex_abs_path)
                                else:
                                    tex = AWDBitmapTexture(AWDBitmapTexture.EXTERNAL, name=self.get_name(state))
                                    tex.url = mc.workspace(pp=tex_abs_path)
                                self.awd.add_texture(tex)
                                self.block_cache.add(state, tex)
                                print('created texture')
 
                            if mat is not None:
                                mat.texture = tex

                        elif state_type == 'place2dTexture' and mat is not None:
                            # Determine from place2dTexture node whether
                            # this material should repeat/wrap
                            rep_uv = mc.getAttr('%s.re' % state)[0]
                            if rep_uv[0] != 1.0 or rep_uv[1] != 1.0:
                                mat.repeat = True
                            elif mc.getAttr(state+'.wu') or mc.getAttr(state+'.wv'):
                                mat.repeat = True

 
 
 
    def sample_materials(self, animated_materials, uvanims):
        idx = 0
        for mat in animated_materials:
            pt = None
            mat_his = mc.listHistory(mat)
            #print('sampling mat', mat)
 
            uvanim = uvanims[idx]
 
            # Find most recent place2DTexture
            for state in mat_his:
                if mc.nodeType(state) == 'place2dTexture':
                    pt = state
                    break
 
            t = mc.getAttr(pt+'.tf')[0]
            #TODO: Don't hard-code duration
            uvanim.add_frame( AWDMatrix2x3([ 1, 0, 0, 1, -t[0], t[1] ]), 40)
 
            idx += 1
 
 
    def export_skeleton(self, root_path):
        skel = AWDSkeleton(name=root_path)
        joints = []
 
        def create_joint(joint_path, world_mtx=None):
            dag_path = self.get_dag_from_path(joint_path)
            tf_fn = om.MFnTransform(dag_path.node())
            tf = tf_fn.transformation()
            joint_wm = tf.asMatrix()
 
            if world_mtx is not None:
                joint_wm = joint_wm * world_mtx
 
            ibm = joint_wm.inverse()
            awd_mtx = self.mtx_maya2awd(ibm)
 
            name = self.get_name(joint_path)
            joint = AWDSkeletonJoint(name=name, inv_bind_mtx=awd_mtx)
 
            self.joint_indices[joint_path] = len(joints)
            print('added joint %s as idx %d' % (joint_path, len(joints)))
            joints.append(name)
 
            children = mc.listRelatives(joint_path, type='joint')
            print('JOINT CHILDREN: %s', str(children))
            if children is not None:
                for child_path in children:
                    joint.add_child_joint( create_joint(child_path, joint_wm) )
 
            return joint
 
        skel.root_joint = create_joint(root_path)
 
        self.awd.add_skeleton(skel)
        self.block_cache.add(root_path, skel)
        self.skeleton_paths.append(root_path)
 
 
 
    def get_skeleton_root(self, joint_path):
        current = joint_path
        parent = mc.listRelatives(current, p=True)
        while parent:
            current = parent
            parent = mc.listRelatives(current, p=True)
 
        if isinstance(current, list):
            current = current[0]
 
        return str(current)
            
 
    def get_dag_from_path(self, path):
        list = om.MSelectionList()
        list.add(path)
        dag_path = om.MDagPath()
        list.getDagPath(0, dag_path, om.MObject())
 
        return dag_path
        
        
 
    def export_mesh_data(self, md, shape_path):
        dag_path = self.get_dag_from_path(shape_path)
        if dag_path.hasFn(om.MFn.kMesh):
            exp_vert_list = []
    
            def get_uvs(vert_it, face_idx):
                us = om.MFloatArray()
                vs = om.MFloatArray()
                uvis = om.MIntArray()
    
                # TODO: Deal with this failing (missing UVs)
                vert_it.getUVs(us, vs, uvis)
                for i in range(len(uvis)):
                    if uvis[i] == face_idx:
                        return (us[i],vs[i])
    
                print('NO UV FOUND!!!!! WHY!!!!!??')
                return (0,0)
    
            def get_vnormal(shape, vert_itx, face_idx):
                vec = om.MVector()
                attr = '%s.vtxFace[%d][%d]' % (shape, vert_itx, face_idx)
                vec = mc.polyNormalPerVertex(attr, q=True, xyz=True)
                return vec
    
            benchmark_start()
    
            print('getting mesh data for %s' % dag_path.fullPathName())
            print('type: %s' % dag_path.node().apiTypeStr())
            vert_it = om.MItMeshVertex(dag_path.node())
            poly_it = om.MItMeshPolygon(dag_path.node())
            while not poly_it.isDone():
                tri_inds = om.MIntArray()
                tri_points = om.MPointArray()
    
                poly_index = poly_it.index()
    
                idx_triple = []
                poly_it.getTriangles(tri_points, tri_inds)
                for i in range(tri_inds.length()):
                    vert_index = tri_inds[i]
    
                    pidx_util = om.MScriptUtil()
                    vert_it.setIndex(vert_index, pidx_util.asIntPtr())
    
                    u,v = get_uvs(vert_it, poly_index)
                    normal = get_vnormal(shape_path, vert_index, poly_index)
                    pos = vert_it.position()

                    exp_vert_list.append(
                        [ vert_index, poly_index, pos[0], pos[1], pos[2], u, v, normal[0], normal[1], normal[2] ])
                    #print('UV %f, %f' % (u,v))
                poly_it.next()

            print('- Raw (expanded) data list created')
            benchmark_print()
    
            # Store this so binding (joint index) data can be
            # put into the right place of the new vertex list
            vert_indices = []
            self.mesh_vert_indices[dag_path.fullPathName()] = vert_indices
    
            vertices = []
            indices = []
            uvs = []
            normals = []
    
            exp_vert_inds = {}
    
            def has_vert(haystack, needle):
                idx = 0
                if needle[0] in exp_vert_inds:
                    for v_idx in exp_vert_inds[needle[0]]:
                        v = haystack[v_idx]
                        correct = True
                        for prop in range(2, 10):
                            if needle[prop] != v[prop]:
                                correct = False
                                break
    
                        idx += 1
    
                return -1
                
            merged_vertices = []
    
            print('- Creating condensed list')
            benchmark_start()
    
            for v in exp_vert_list:
                idx = has_vert(merged_vertices, v)
                if idx >= 0:
                    # Already has vertex
                    indices.append(idx)
                else:
                    # Store this for binding data
                    vert_indices.append(v[0])
    
                    # This vertex will be added into the expanded list of vertices,
                    # which can get very large. To enable fast look-up, we map it's
                    # original index to that in the expanded list
                    vert_index = v[0]
                    if vert_index not in exp_vert_inds:
                        exp_vert_inds[vert_index] = []
                    exp_vert_inds[vert_index].append(len(merged_vertices))
    
                    indices.append(len(merged_vertices))
                    merged_vertices.append(v)
                
            for v in merged_vertices:
                # Add vertex and index
                vertices.append(v[2])   # X
                vertices.append(v[3])   # Y
                vertices.append(-v[4])  # Z (inverted)
                uvs.append(v[5])        # U
                uvs.append(1-v[6])      # V
                normals.append(v[7])    # Normal X
                normals.append(v[8])    # Normal Y
                normals.append(-v[9])   # Normal Z (inverted)
    
            benchmark_print()
            print('- DONE! Flipping windings')
            benchmark_start()
    
            # Flip windings
            for idx in range(1, len(indices), 3):
                tmp = indices[idx]
                indices[idx] = indices[idx+1]
                indices[idx+1] = tmp
    
            benchmark_print()
            print('- Creating sub-mesh')
            sub = AWDSubGeom()
            sub.add_stream(pyawd.geom.STR_VERTICES, vertices)
            sub.add_stream(pyawd.geom.STR_TRIANGLES, indices)
            sub.add_stream(pyawd.geom.STR_UVS, uvs)
            sub.add_stream(pyawd.geom.STR_VERTEX_NORMALS, normals)
    
            print('- Adding sub-mesh')
    
            md.add_sub_geom(sub)
    
            # Store mesh data block to block cache

    def set_attributes(self, dag_path, awd_elem):
        if self.include_attr:
            extra_attributes = mc.listAttr(dag_path, ud=True)
            if extra_attributes is not None:
                for attr in extra_attributes:
                    val = mc.getAttr('%s.%s' % (dag_path, attr))
                    awd_elem.attributes[self.user_ns][str(attr)] = val

    def get_name(self, dag_path):
        # TODO: Deal with unicode names. In pyawd?
        return str(dag_path.split('|')[-1])
        
    def mtx_list2awd(self, mtx):
        mtx_list = [1,0,0,0,1,0,0,0,1,0,0,0]
        mtx_list[0] = mtx[0]
        mtx_list[1] = mtx[1]
        mtx_list[2] = -mtx[2]

        mtx_list[3] = mtx[4]
        mtx_list[4] = mtx[5]
        mtx_list[5] = -mtx[6]

        mtx_list[6] = -mtx[8]
        mtx_list[7] = -mtx[9]
        mtx_list[8] = mtx[10]

        mtx_list[9] = mtx[12]
        mtx_list[10] = mtx[13]
        mtx_list[11] = -mtx[14]
 
        return AWDMatrix3x4(mtx_list)
        
    def mtx_maya2awd(self, mtx):
        mtx_list = []
        for i in range(16):
           row_idx = math.floor(i/4)
           col_idx = i%4
           mtx_list.append(mtx(int(row_idx), int(col_idx)))
 
        #mtx_list[1] *= -1
        #mtx_list[2] *= -1
        #mtx_list[3] *= -1
        #mtx_list[4] *= -1
        #mtx_list[8] *= -1
 
        #print(mtx_list[0:4])
        #print(mtx_list[4:8])
        #print(mtx_list[8:12])
        #print(mtx_list[12:])
        return self.mtx_list2awd(mtx_list)
        
