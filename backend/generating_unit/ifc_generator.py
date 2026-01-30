import ifcopenshell
import ifcopenshell.api
import ifcopenshell.guid
import time
import uuid

class IfcGenerator:
    def __init__(self, project_name="ConstructionProject"):
        # Create a blank model with IFC4 schema for better compatibility with modern viewers
        self.model = ifcopenshell.file(schema="IFC4")
        
        # Create Project hierarchy
        self.project = ifcopenshell.api.run("root.create_entity", self.model, ifc_class="IfcProject", name=project_name)
        
        # Default units (SI)
        ifcopenshell.api.run("unit.assign_unit", self.model)
        
        # Create context with explicit coordinate system
        self.context = ifcopenshell.api.run("context.add_context", self.model, context_type="Model")
        
        # Create Body context for geometry
        self.body_context = ifcopenshell.api.run("context.add_context", self.model, context_type="Model", 
                                                 context_identifier="Body", target_view="MODEL_VIEW", parent=self.context)

        # Add a default material
        self.material = ifcopenshell.api.run("material.add_material", self.model, name="Concrete")
        self.material_set = ifcopenshell.api.run("material.assign_material", self.model, 
                                                 products=[], type="IFCMATERIALLAYERSET", material=self.material)

        # Create Site, Building, Storey
        self.site = ifcopenshell.api.run("root.create_entity", self.model, ifc_class="IfcSite", name="MySite")
        self.building = ifcopenshell.api.run("root.create_entity", self.model, ifc_class="IfcBuilding", name="MyBuilding")
        self.storey = ifcopenshell.api.run("root.create_entity", self.model, ifc_class="IfcBuildingStorey", name="Level 1")
        
        # Set Storey Elevation
        self.storey.Elevation = 0.0

        # Assign Placements to hierarchy (Critical for rendering)
        identity_matrix = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ]
        ifcopenshell.api.run("geometry.edit_object_placement", self.model, products=[self.site], matrix=identity_matrix)
        ifcopenshell.api.run("geometry.edit_object_placement", self.model, products=[self.building], matrix=identity_matrix)
        ifcopenshell.api.run("geometry.edit_object_placement", self.model, products=[self.storey], matrix=identity_matrix)

        # Assign hierarchy
        ifcopenshell.api.run("aggregate.assign_object", self.model, relating_object=self.project, products=[self.site])
        ifcopenshell.api.run("aggregate.assign_object", self.model, relating_object=self.site, products=[self.building])
        ifcopenshell.api.run("aggregate.assign_object", self.model, relating_object=self.building, products=[self.storey])
        
        print(f"IFC Hierarchy initialized (IFC4): Project -> Site -> Building -> Storey")

    def create_column(self, x: float, y: float, width: float, depth: float, height: float, elevation: float = 0.0):
        """
        Create a rectangular column at (x, y).
        """
        # Create the column element
        column = ifcopenshell.api.run("root.create_entity", self.model, ifc_class="IfcColumn", name="Column")
        
        # Define representation (Geometry)
        # Create a 2D profile for extrusion
        profile = self.model.createIfcRectangleProfileDef(ProfileType="AREA", XDim=width, YDim=depth)
        
        # Create extrusion
        # Position of the profile is (0,0) relative to the column placement
        # We need to place the column in the world
        
        # Create the 3D representation
        representation = ifcopenshell.api.run("geometry.add_profile_representation", self.model, 
                                              context=self.body_context, profile=profile, depth=height)
        
        # Assign representation to column
        ifcopenshell.api.run("geometry.assign_representation", self.model, products=[column], representation=representation)
        
        # Place the column
        # Matrix is [x, y, z]
        matrix = [
            [1.0, 0.0, 0.0, x],
            [0.0, 1.0, 0.0, y],
            [0.0, 0.0, 1.0, elevation],
            [0.0, 0.0, 0.0, 1.0]
        ]
        
        ifcopenshell.api.run("geometry.edit_object_placement", self.model, products=[column], matrix=matrix)
        
        # Assign material
        ifcopenshell.api.run("material.assign_material", self.model, products=[column], material=self.material)
        
        # Assign to storey
        ifcopenshell.api.run("spatial.assign_container", self.model, relating_structure=self.storey, products=[column])
        
        return column

    def create_beam(self, x1: float, y1: float, x2: float, y2: float, width: float, depth: float, elevation: float):
        """
        Create a beam connecting (x1, y1) and (x2, y2) at a specific elevation (z).
        """
        beam = ifcopenshell.api.run("root.create_entity", self.model, ifc_class="IfcBeam", name="Beam")
        
        # Calculate length and rotation
        import math
        dx = x2 - x1
        dy = y2 - y1
        length = math.sqrt(dx*dx + dy*dy)
        rotation = math.atan2(dy, dx)
        
        # Profile (Beam cross section: depth is height of beam, width is width)
        profile = self.model.createIfcRectangleProfileDef(ProfileType="AREA", XDim=length, YDim=width)
        
        # Representation (Extrude along Z by the 'depth' of the beam)
        # Note: In IFC, we can extrude a profile along an axis.
        # To make a beam, we can extrude the rectangular cross-section along the length.
        # For simplicity, we'll extrude the (length x width) profile by 'depth' (height of beam).
        representation = ifcopenshell.api.run("geometry.add_profile_representation", self.model, 
                                              context=self.body_context, profile=profile, depth=depth)
        
        ifcopenshell.api.run("geometry.assign_representation", self.model, products=[beam], representation=representation)
        
        # Placement
        # We need to place the center of the beam at the midpoint and rotate it.
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        
        # Rotation Matrix around Z
        cos_r = math.cos(rotation)
        sin_r = math.sin(rotation)
        
        # Matrix:
        # [ cos -sin  0  cx ]
        # [ sin  cos  0  cy ]
        # [  0    0   1  el ]
        # [  0    0   0   1 ]
        matrix = [
            [cos_r, -sin_r, 0.0, cx],
            [sin_r,  cos_r, 0.0, cy],
            [0.0,    0.0,   1.0, elevation],
            [0.0,    0.0,   0.0, 1.0]
        ]
        
        ifcopenshell.api.run("geometry.edit_object_placement", self.model, products=[beam], matrix=matrix)
        
        # Assign material
        ifcopenshell.api.run("material.assign_material", self.model, products=[beam], material=self.material)
        
        ifcopenshell.api.run("spatial.assign_container", self.model, relating_structure=self.storey, products=[beam])
        
        return beam

    def create_slab(self, x: float, y: float, width: float, depth: float, thickness: float, elevation: float):
        """
        Create a rectangular slab.
        """
        slab = ifcopenshell.api.run("root.create_entity", self.model, ifc_class="IfcSlab", name="Slab")
        
        profile = self.model.createIfcRectangleProfileDef(ProfileType="AREA", XDim=width, YDim=depth)
        representation = ifcopenshell.api.run("geometry.add_profile_representation", self.model, 
                                              context=self.body_context, profile=profile, depth=thickness)
        
        ifcopenshell.api.run("geometry.assign_representation", self.model, products=[slab], representation=representation)
        
        matrix = [
            [1.0, 0.0, 0.0, x],
            [0.0, 1.0, 0.0, y],
            [0.0, 0.0, 1.0, elevation],
            [0.0, 0.0, 0.0, 1.0]
        ]
        
        ifcopenshell.api.run("geometry.edit_object_placement", self.model, products=[slab], matrix=matrix)
        
        # Assign material
        ifcopenshell.api.run("material.assign_material", self.model, products=[slab], material=self.material)
        
        ifcopenshell.api.run("spatial.assign_container", self.model, relating_structure=self.storey, products=[slab])
        
        return slab

    def create_generic_element(self, x: float, y: float, width: float, depth: float, height: float, elevation: float, ifc_class="IfcBuildingElementProxy", name="Element"):
        """
        Create a generic rectangular element for objects like Doors/Windows/Slabs
        to ensure they appear in the 3D viewer.
        """
        element = ifcopenshell.api.run("root.create_entity", self.model, ifc_class=ifc_class, name=name)
        
        # Profile
        profile = self.model.createIfcRectangleProfileDef(ProfileType="AREA", XDim=width, YDim=depth)
        
        # Representation (Extrusion)
        representation = ifcopenshell.api.run("geometry.add_profile_representation", self.model, 
                                              context=self.body_context, profile=profile, depth=height)
        
        ifcopenshell.api.run("geometry.assign_representation", self.model, products=[element], representation=representation)
        
        # Placement
        # Fix: Manually construct the matrix instead of using the API if it's missing in this version
        # 4x4 Identity Matrix with translation
        # [ 1, 0, 0, x ]
        # [ 0, 1, 0, y ]
        # [ 0, 0, 1, z ]
        # [ 0, 0, 0, 1 ]
        
        # However, ifcopenshell.api.run("geometry.edit_object_placement") expects a certain format or 
        # we can just use "geometry.edit_object_placement" with a dictionary if supported, 
        # or creating the placement manually.
        
        # Let's try to see if we can just pass the matrix directly to edit_object_placement if it supports it, 
        # OR use a simpler way: create placement manually.
        
        # Correct approach for 0.8.4+ if calculate_matrix is missing:
        # Just create the local placement.
        
        # But wait, edit_object_placement usually takes a matrix.
        # If calculate_matrix is missing, we can provide the matrix as a nested list.
        matrix = [
            [1.0, 0.0, 0.0, x],
            [0.0, 1.0, 0.0, y],
            [0.0, 0.0, 1.0, elevation],
            [0.0, 0.0, 0.0, 1.0]
        ]
        
        ifcopenshell.api.run("geometry.edit_object_placement", self.model, products=[element], matrix=matrix)
        
        # Assign material
        ifcopenshell.api.run("material.assign_material", self.model, products=[element], material=self.material)
        
        # Assign to storey
        ifcopenshell.api.run("spatial.assign_container", self.model, relating_structure=self.storey, products=[element])
        return element

    def generate_simple_extrusion(self, det_results: dict, scale: float, height: float, floor_count: int):
        """
        Mode 1: Simple Rule-Based Extrusion (Baseline).
        Iterates through detections and extrudes them vertically.
        """
        detections = det_results.get('detections', [])
        if not detections:
            print("No detections found for IFC generation.")
            return

        # 1. First pass: Calculate average center to offset the model to (0,0,0)
        # This ensures the model is visible in the center of the 3D viewer.
        total_cx = 0
        total_cy = 0
        for det in detections:
            bbox = det['bbox']
            x_mid = (bbox[0] + bbox[2]) / 2 * scale
            y_mid = (bbox[1] + bbox[3]) / 2 * scale
            total_cx += x_mid
            total_cy += y_mid
        
        offset_x = total_cx / len(detections)
        offset_y = total_cy / len(detections)

        # 2. Second pass: Create elements with offset
        for det in detections:
            cls = det['class'].lower() 
            bbox = det['bbox']
            
            x1_m = bbox[0] * scale
            y1_m = bbox[1] * scale
            x2_m = bbox[2] * scale
            y2_m = bbox[3] * scale
            
            width = x2_m - x1_m
            depth = y2_m - y1_m
            
            # Centered at (0,0) relative to the whole building
            cx = (x1_m + width / 2) - offset_x
            cy = -((y1_m + depth / 2) - offset_y) # Flip Y and apply offset
            
            for i in range(floor_count):
                elevation = i * height
                
                # Logic for different classes
                if cls in ['column', 'person']: 
                    self.create_column(cx, cy, width, depth, height, elevation=elevation)
                elif cls in ['door', 'double-door', 'sliding door', 'garage door']:
                    self.create_generic_element(cx, cy, width, depth, height * 0.8, elevation, ifc_class="IfcDoor", name=cls.title())
                elif cls in ['window', 'ventilator']:
                    sill_height = height * 0.3
                    win_height = height * 0.4
                    self.create_generic_element(cx, cy, width, depth, win_height, elevation + sill_height, ifc_class="IfcWindow", name=cls.title())
                elif cls in ['staircase', 'stairs']:
                    self.create_generic_element(cx, cy, width, depth, height * 0.5, elevation, ifc_class="IfcStair", name="Staircase")
                elif cls == 'slab':
                     self.create_slab(cx, cy, width, depth, 0.2, elevation)
                else:
                    self.create_generic_element(cx, cy, width, depth, height * 0.5, elevation, ifc_class="IfcBuildingElementProxy", name=cls.title())
        
        print(f"Generated simple extrusion for {len(detections)} objects across {floor_count} floors.")

    def generate_advanced_structure(self, graph_data: dict, scale: float, height: float, floor_count: int):
        """
        Mode 2: Advanced GNN-based Reconstruction.
        Uses graph data (nodes and edges) to generate a connected structure.
        """
        nodes = graph_data.get('nodes', [])
        if not nodes:
            print("No nodes found for advanced structure generation.")
            return

        # 1. Calculate offset to center the model
        total_cx = sum(node['x'] for node in nodes) * scale
        total_cy = sum(node['y'] for node in nodes) * scale
        offset_x = total_cx / len(nodes)
        offset_y = total_cy / len(nodes)

        # 2. Create Nodes (Columns)
        node_map = {}
        for i, node in enumerate(nodes):
            cx = (node['x'] * scale) - offset_x
            cy = -((node['y'] * scale) - offset_y) # Flip Y
            width = node['width'] * scale
            depth = node['depth'] * scale
            
            for f in range(floor_count):
                elevation = f * height
                self.create_column(cx, cy, width, depth, height, elevation=elevation)
                if f == 0:
                    node_map[i] = (cx, cy)

        # 3. Create Edges (Beams)
        edges = graph_data.get('edges', [])
        for edge in edges:
            source_idx = edge['source']
            target_idx = edge['target']
            
            if source_idx in node_map and target_idx in node_map:
                p1 = node_map[source_idx]
                p2 = node_map[target_idx]
                
                beam_width = 0.3
                beam_depth = 0.5 
                
                for f in range(floor_count):
                    elevation = (f + 1) * height - beam_depth 
                    self.create_beam(p1[0], p1[1], p2[0], p2[1], beam_width, beam_depth, elevation=elevation)
        
        print(f"Generated advanced structure with {len(nodes)} nodes and {len(edges)} edges.")

    def save(self, path: str):
        self.model.write(path)
