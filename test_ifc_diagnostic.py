"""
Comprehensive IFC Generation and Validation Diagnostic Script
=============================================================
This script tests the entire IFC generation pipeline and provides detailed diagnostics.

Usage:
    python test_ifc_diagnostic.py

Requirements:
    - ifcopenshell
    - numpy
    - requests (for API testing)
    - Optional: ifcpatch, ifcdiff for advanced validation
"""

import os
import sys
import json
import logging
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Configure logging
log_filename = f"ifc_diagnostic_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def print_header(message: str):
    """Print a formatted header"""
    logger.info(f"\n{'='*80}")
    logger.info(f"{Colors.BOLD}{Colors.BLUE}{message}{Colors.RESET}")
    logger.info(f"{'='*80}\n")

def print_success(message: str):
    """Print success message"""
    logger.info(f"{Colors.GREEN}✓ {message}{Colors.RESET}")

def print_error(message: str):
    """Print error message"""
    logger.error(f"{Colors.RED}✗ {message}{Colors.RESET}")

def print_warning(message: str):
    """Print warning message"""
    logger.warning(f"{Colors.YELLOW}⚠ {message}{Colors.RESET}")

def print_info(message: str):
    """Print info message"""
    logger.info(f"{Colors.BLUE}ℹ {message}{Colors.RESET}")


class IFCDiagnosticTester:
    """Comprehensive IFC generation and validation tester"""
    
    def __init__(self):
        self.results = {
            'tests_passed': [],
            'tests_failed': [],
            'warnings': [],
            'ifc_files_generated': [],
            'timestamp': datetime.now().isoformat()
        }
        self.output_dir = Path('diagnostic_output')
        self.output_dir.mkdir(exist_ok=True)
        
    def run_all_tests(self):
        """Run all diagnostic tests"""
        print_header("IFC GENERATION & VALIDATION DIAGNOSTIC SUITE")
        
        # Test 1: Check dependencies
        self.test_dependencies()
        
        # Test 2: Test basic IFC creation
        self.test_basic_ifc_creation()
        
        # Test 3: Test geometry generation
        self.test_geometry_generation()
        
        # Test 4: Test complete building structure
        self.test_complete_building()
        
        # Test 5: Validate IFC files
        self.test_ifc_validation()
        
        # Test 6: Test file size and complexity
        self.test_file_characteristics()
        
        # Test 7: Test coordinate system
        self.test_coordinate_system()
        
        # Test 8: Test backend API (if running)
        self.test_backend_api()
        
        # Generate report
        self.generate_report()
        
    def test_dependencies(self):
        """Test if all required dependencies are available"""
        print_header("TEST 1: Checking Dependencies")
        
        dependencies = {
            'ifcopenshell': None,
            'numpy': None,
            'requests': None,
        }
        
        for dep_name in dependencies.keys():
            try:
                module = __import__(dep_name)
                dependencies[dep_name] = getattr(module, '__version__', 'installed')
                print_success(f"{dep_name}: {dependencies[dep_name]}")
                self.results['tests_passed'].append(f"Dependency {dep_name} available")
            except ImportError as e:
                print_error(f"{dep_name}: NOT INSTALLED - {str(e)}")
                self.results['tests_failed'].append(f"Dependency {dep_name} missing")
        
        # Check optional dependencies
        optional_deps = ['ifcpatch', 'ifcdiff']
        for dep_name in optional_deps:
            try:
                __import__(dep_name)
                print_success(f"{dep_name} (optional): installed")
            except ImportError:
                print_warning(f"{dep_name} (optional): not installed")
        
        return all(v is not None for v in dependencies.values())
    
    def test_basic_ifc_creation(self):
        """Test basic IFC file creation"""
        print_header("TEST 2: Basic IFC File Creation")
        
        try:
            import ifcopenshell
            import ifcopenshell.api
            
            # Create minimal IFC file
            print_info("Creating minimal IFC4 file...")
            ifc_file = ifcopenshell.api.run("project.create_file", version="IFC4")
            
            # Create project
            print_info("Creating IFC Project...")
            project = ifcopenshell.api.run("root.create_entity", ifc_file, ifc_class="IfcProject", name="Test Project")
            
            # Create units
            print_info("Setting up units...")
            ifcopenshell.api.run("unit.assign_unit", ifc_file, length={"is_metric": True, "raw": "METERS"})
            
            # Create site
            print_info("Creating site...")
            site = ifcopenshell.api.run("root.create_entity", ifc_file, ifc_class="IfcSite", name="Test Site")
            ifcopenshell.api.run("aggregate.assign_object", ifc_file, relating_object=project, products=[site])
            
            # Save file
            output_file = self.output_dir / "test_basic.ifc"
            ifc_file.write(str(output_file))
            print_success(f"Basic IFC file created: {output_file}")
            
            # Verify file
            file_size = output_file.stat().st_size
            print_info(f"File size: {file_size} bytes")
            
            # Try to reopen it
            print_info("Verifying file can be reopened...")
            reopened = ifcopenshell.open(str(output_file))
            entities_count = len(list(reopened))
            print_success(f"File reopened successfully. Contains {entities_count} entities")
            
            self.results['tests_passed'].append("Basic IFC creation")
            self.results['ifc_files_generated'].append(str(output_file))
            
            return True, str(output_file)
            
        except Exception as e:
            print_error(f"Basic IFC creation failed: {str(e)}")
            logger.error(traceback.format_exc())
            self.results['tests_failed'].append(f"Basic IFC creation: {str(e)}")
            return False, None
    
    def test_geometry_generation(self):
        """Test geometry creation (columns, beams, slabs)"""
        print_header("TEST 3: Geometry Generation")
        
        try:
            import ifcopenshell
            import ifcopenshell.api
            import ifcopenshell.geom
            import numpy as np
            
            print_info("Creating IFC file with geometry...")
            ifc_file = ifcopenshell.api.run("project.create_file", version="IFC4")
            project = ifcopenshell.api.run("root.create_entity", ifc_file, ifc_class="IfcProject", name="Geometry Test")
            
            # Set up context
            print_info("Setting up geometric context...")
            context = ifcopenshell.api.run("context.add_context", ifc_file, context_type="Model")
            body = ifcopenshell.api.run(
                "context.add_context",
                ifc_file,
                context_type="Model",
                context_identifier="Body",
                target_view="MODEL_VIEW",
                parent=context
            )
            
            # Set up units
            ifcopenshell.api.run("unit.assign_unit", ifc_file, length={"is_metric": True, "raw": "METERS"})
            
            # Create site and building
            site = ifcopenshell.api.run("root.create_entity", ifc_file, ifc_class="IfcSite", name="Site")
            ifcopenshell.api.run("aggregate.assign_object", ifc_file, relating_object=project, products=[site])
            
            building = ifcopenshell.api.run("root.create_entity", ifc_file, ifc_class="IfcBuilding", name="Building")
            ifcopenshell.api.run("aggregate.assign_object", ifc_file, relating_object=site, products=[building])
            
            storey = ifcopenshell.api.run("root.create_entity", ifc_file, ifc_class="IfcBuildingStorey", name="Ground Floor")
            ifcopenshell.api.run("aggregate.assign_object", ifc_file, relating_object=building, products=[storey])
            
            # Test 1: Create a column
            print_info("Creating column geometry...")
            column = ifcopenshell.api.run("root.create_entity", ifc_file, ifc_class="IfcColumn", name="Test Column")
            ifcopenshell.api.run("spatial.assign_container", ifc_file, relating_structure=storey, products=[column])
            
            # Create rectangular profile for column
            size = ifc_file.createIfcPositiveLengthMeasure(0.3)  # 300mm x 300mm
            profile = ifc_file.createIfcRectangleProfileDef(
                "AREA",
                "300x300",
                None,
                size,
                size
            )
            
            # Create extrusion
            height = ifc_file.createIfcPositiveLengthMeasure(3.0)  # 3m high
            direction = ifc_file.createIfcDirection((0., 0., 1.))
            extrusion = ifc_file.createIfcExtrudedAreaSolid(
                profile,
                None,
                direction,
                height
            )
            
            # Create representation
            representation = ifc_file.createIfcShapeRepresentation(
                body,
                "Body",
                "SweptSolid",
                [extrusion]
            )
            
            product_shape = ifc_file.createIfcProductDefinitionShape(None, None, [representation])
            column.Representation = product_shape
            
            # Set column placement at origin
            origin = ifc_file.createIfcCartesianPoint((0., 0., 0.))
            placement = ifc_file.createIfcLocalPlacement(None, ifc_file.createIfcAxis2Placement3D(origin))
            column.ObjectPlacement = placement
            
            print_success("Column created successfully")
            
            # Test 2: Create a beam
            print_info("Creating beam geometry...")
            beam = ifcopenshell.api.run("root.create_entity", ifc_file, ifc_class="IfcBeam", name="Test Beam")
            ifcopenshell.api.run("spatial.assign_container", ifc_file, relating_structure=storey, products=[beam])
            
            # Rectangular profile for beam
            beam_width = ifc_file.createIfcPositiveLengthMeasure(0.2)
            beam_height = ifc_file.createIfcPositiveLengthMeasure(0.4)
            beam_profile = ifc_file.createIfcRectangleProfileDef(
                "AREA",
                "200x400",
                None,
                beam_width,
                beam_height
            )
            
            beam_length = ifc_file.createIfcPositiveLengthMeasure(5.0)  # 5m long
            beam_direction = ifc_file.createIfcDirection((1., 0., 0.))  # Along X-axis
            beam_extrusion = ifc_file.createIfcExtrudedAreaSolid(
                beam_profile,
                None,
                beam_direction,
                beam_length
            )
            
            beam_representation = ifc_file.createIfcShapeRepresentation(
                body,
                "Body",
                "SweptSolid",
                [beam_extrusion]
            )
            
            beam_shape = ifc_file.createIfcProductDefinitionShape(None, None, [beam_representation])
            beam.Representation = beam_shape
            
            # Place beam at column top
            beam_origin = ifc_file.createIfcCartesianPoint((0., 0., 3.0))
            beam_placement = ifc_file.createIfcLocalPlacement(None, ifc_file.createIfcAxis2Placement3D(beam_origin))
            beam.ObjectPlacement = beam_placement
            
            print_success("Beam created successfully")
            
            # Test 3: Create a slab
            print_info("Creating slab geometry...")
            slab = ifcopenshell.api.run("root.create_entity", ifc_file, ifc_class="IfcSlab", name="Test Slab")
            ifcopenshell.api.run("spatial.assign_container", ifc_file, relating_structure=storey, products=[slab])
            
            # Create rectangular profile for slab (5m x 5m)
            slab_points = [
                ifc_file.createIfcCartesianPoint((0., 0.)),
                ifc_file.createIfcCartesianPoint((5., 0.)),
                ifc_file.createIfcCartesianPoint((5., 5.)),
                ifc_file.createIfcCartesianPoint((0., 5.)),
                ifc_file.createIfcCartesianPoint((0., 0.))
            ]
            slab_polyline = ifc_file.createIfcPolyline(slab_points)
            slab_profile = ifc_file.createIfcArbitraryClosedProfileDef("AREA", None, slab_polyline)
            
            # Extrude slab thickness (200mm)
            slab_thickness = ifc_file.createIfcPositiveLengthMeasure(0.2)
            slab_direction = ifc_file.createIfcDirection((0., 0., 1.))
            slab_extrusion = ifc_file.createIfcExtrudedAreaSolid(
                slab_profile,
                None,
                slab_direction,
                slab_thickness
            )
            
            slab_representation = ifc_file.createIfcShapeRepresentation(
                body,
                "Body",
                "SweptSolid",
                [slab_extrusion]
            )
            
            slab_shape = ifc_file.createIfcProductDefinitionShape(None, None, [slab_representation])
            slab.Representation = slab_shape
            
            # Place slab at floor level
            slab_origin = ifc_file.createIfcCartesianPoint((0., 0., 0.))
            slab_placement = ifc_file.createIfcLocalPlacement(None, ifc_file.createIfcAxis2Placement3D(slab_origin))
            slab.ObjectPlacement = slab_placement
            
            print_success("Slab created successfully")
            
            # Save file
            output_file = self.output_dir / "test_geometry.ifc"
            ifc_file.write(str(output_file))
            print_success(f"Geometry test file created: {output_file}")
            
            # Verify geometry
            print_info("Verifying geometry...")
            settings = ifcopenshell.geom.settings()
            iterator = ifcopenshell.geom.iterator(settings, ifc_file, multiprocessing.cpu_count())
            
            geometry_count = 0
            if iterator.initialize():
                while True:
                    shape = iterator.get()
                    geometry_count += 1
                    element = ifc_file.by_id(shape.id)
                    print_info(f"  - {element.is_a()}: {element.Name or 'Unnamed'}")
                    if not iterator.next():
                        break
            
            print_success(f"Successfully generated geometry for {geometry_count} elements")
            
            self.results['tests_passed'].append("Geometry generation")
            self.results['ifc_files_generated'].append(str(output_file))
            
            return True, str(output_file)
            
        except Exception as e:
            print_error(f"Geometry generation failed: {str(e)}")
            logger.error(traceback.format_exc())
            self.results['tests_failed'].append(f"Geometry generation: {str(e)}")
            return False, None
    
    def test_complete_building(self):
        """Test creation of a complete building structure"""
        print_header("TEST 4: Complete Building Structure")
        
        try:
            import ifcopenshell
            import ifcopenshell.api
            
            print_info("Creating complete building structure...")
            ifc_file = ifcopenshell.api.run("project.create_file", version="IFC4")
            
            # Project setup
            project = ifcopenshell.api.run("root.create_entity", ifc_file, 
                                          ifc_class="IfcProject", 
                                          name="Complete Building Test")
            
            # Units
            ifcopenshell.api.run("unit.assign_unit", ifc_file, 
                                length={"is_metric": True, "raw": "METERS"})
            
            # Context
            context = ifcopenshell.api.run("context.add_context", ifc_file, context_type="Model")
            body = ifcopenshell.api.run("context.add_context", ifc_file,
                                       context_type="Model",
                                       context_identifier="Body",
                                       target_view="MODEL_VIEW",
                                       parent=context)
            
            # Spatial hierarchy
            print_info("Creating spatial hierarchy...")
            site = ifcopenshell.api.run("root.create_entity", ifc_file, 
                                       ifc_class="IfcSite", 
                                       name="Construction Site")
            ifcopenshell.api.run("aggregate.assign_object", ifc_file, 
                                relating_object=project, products=[site])
            
            building = ifcopenshell.api.run("root.create_entity", ifc_file,
                                           ifc_class="IfcBuilding",
                                           name="Main Building")
            ifcopenshell.api.run("aggregate.assign_object", ifc_file,
                                relating_object=site, products=[building])
            
            # Create multiple storeys
            storeys = []
            for i in range(3):  # 3 floors
                storey = ifcopenshell.api.run("root.create_entity", ifc_file,
                                             ifc_class="IfcBuildingStorey",
                                             name=f"Floor {i+1}")
                ifcopenshell.api.run("aggregate.assign_object", ifc_file,
                                    relating_object=building, products=[storey])
                
                # Set elevation
                origin = ifc_file.createIfcCartesianPoint((0., 0., i * 3.5))
                placement = ifc_file.createIfcLocalPlacement(
                    None,
                    ifc_file.createIfcAxis2Placement3D(origin)
                )
                storey.ObjectPlacement = placement
                storey.Elevation = i * 3.5
                
                storeys.append(storey)
                print_info(f"  Created Floor {i+1} at elevation {i * 3.5}m")
            
            # Create columns grid (3x3)
            print_info("Creating column grid...")
            column_count = 0
            for i in range(3):
                for j in range(3):
                    for storey_idx, storey in enumerate(storeys):
                        column = ifcopenshell.api.run("root.create_entity", ifc_file,
                                                     ifc_class="IfcColumn",
                                                     name=f"C-{i}{j}-F{storey_idx+1}")
                        ifcopenshell.api.run("spatial.assign_container", ifc_file,
                                           relating_structure=storey, products=[column])
                        
                        # Position column
                        x_pos = i * 5.0  # 5m grid
                        y_pos = j * 5.0
                        z_pos = storey_idx * 3.5
                        
                        origin = ifc_file.createIfcCartesianPoint((x_pos, y_pos, 0.))
                        rel_placement = storey.ObjectPlacement if storey_idx > 0 else None
                        placement = ifc_file.createIfcLocalPlacement(
                            rel_placement,
                            ifc_file.createIfcAxis2Placement3D(origin)
                        )
                        column.ObjectPlacement = placement
                        column_count += 1
            
            print_success(f"Created {column_count} columns")
            
            # Create beams connecting columns
            print_info("Creating beams...")
            beam_count = 0
            for storey_idx, storey in enumerate(storeys):
                # Beams in X direction
                for j in range(3):
                    for i in range(2):
                        beam = ifcopenshell.api.run("root.create_entity", ifc_file,
                                                   ifc_class="IfcBeam",
                                                   name=f"B-X{i}{j}-F{storey_idx+1}")
                        ifcopenshell.api.run("spatial.assign_container", ifc_file,
                                           relating_structure=storey, products=[beam])
                        beam_count += 1
                
                # Beams in Y direction
                for i in range(3):
                    for j in range(2):
                        beam = ifcopenshell.api.run("root.create_entity", ifc_file,
                                                   ifc_class="IfcBeam",
                                                   name=f"B-Y{i}{j}-F{storey_idx+1}")
                        ifcopenshell.api.run("spatial.assign_container", ifc_file,
                                           relating_structure=storey, products=[beam])
                        beam_count += 1
            
            print_success(f"Created {beam_count} beams")
            
            # Create slabs
            print_info("Creating slabs...")
            slab_count = 0
            for storey_idx, storey in enumerate(storeys):
                slab = ifcopenshell.api.run("root.create_entity", ifc_file,
                                           ifc_class="IfcSlab",
                                           name=f"Slab-F{storey_idx+1}")
                ifcopenshell.api.run("spatial.assign_container", ifc_file,
                                   relating_structure=storey, products=[slab])
                slab_count += 1
            
            print_success(f"Created {slab_count} slabs")
            
            # Save file
            output_file = self.output_dir / "test_complete_building.ifc"
            ifc_file.write(str(output_file))
            print_success(f"Complete building file created: {output_file}")
            
            # Statistics
            total_entities = len(list(ifc_file))
            print_info(f"Total IFC entities: {total_entities}")
            print_info(f"Columns: {column_count}")
            print_info(f"Beams: {beam_count}")
            print_info(f"Slabs: {slab_count}")
            print_info(f"Storeys: {len(storeys)}")
            
            self.results['tests_passed'].append("Complete building structure")
            self.results['ifc_files_generated'].append(str(output_file))
            
            return True, str(output_file)
            
        except Exception as e:
            print_error(f"Complete building creation failed: {str(e)}")
            logger.error(traceback.format_exc())
            self.results['tests_failed'].append(f"Complete building: {str(e)}")
            return False, None
    
    def test_ifc_validation(self):
        """Validate generated IFC files"""
        print_header("TEST 5: IFC File Validation")
        
        if not self.results['ifc_files_generated']:
            print_warning("No IFC files to validate")
            return False
        
        try:
            import ifcopenshell
            import ifcopenshell.validate
            
            all_valid = True
            for ifc_path in self.results['ifc_files_generated']:
                print_info(f"Validating: {ifc_path}")
                
                try:
                    # Open file
                    ifc_file = ifcopenshell.open(ifc_path)
                    
                    # Check schema
                    schema = ifc_file.schema
                    print_info(f"  Schema: {schema}")
                    
                    # Count entities
                    entity_count = len(list(ifc_file))
                    print_info(f"  Total entities: {entity_count}")
                    
                    # Check for required elements
                    projects = ifc_file.by_type("IfcProject")
                    sites = ifc_file.by_type("IfcSite")
                    buildings = ifc_file.by_type("IfcBuilding")
                    
                    print_info(f"  Projects: {len(projects)}")
                    print_info(f"  Sites: {len(sites)}")
                    print_info(f"  Buildings: {len(buildings)}")
                    
                    # Check for geometry
                    columns = ifc_file.by_type("IfcColumn")
                    beams = ifc_file.by_type("IfcBeam")
                    slabs = ifc_file.by_type("IfcSlab")
                    
                    print_info(f"  Columns: {len(columns)}")
                    print_info(f"  Beams: {len(beams)}")
                    print_info(f"  Slabs: {len(slabs)}")
                    
                    # Validate structure
                    if len(projects) == 0:
                        print_warning("  No IfcProject found!")
                        all_valid = False
                    
                    # Try validation (if available)
                    try:
                        validation_result = ifcopenshell.validate.validate(ifc_file)
                        if validation_result:
                            print_success(f"  File is valid")
                        else:
                            print_warning(f"  File has validation issues")
                            all_valid = False
                    except AttributeError:
                        print_info("  Advanced validation not available")
                    
                    print_success(f"Basic validation passed for {Path(ifc_path).name}")
                    
                except Exception as e:
                    print_error(f"  Validation failed: {str(e)}")
                    all_valid = False
            
            if all_valid:
                self.results['tests_passed'].append("IFC validation")
            else:
                self.results['warnings'].append("Some IFC files have validation issues")
            
            return all_valid
            
        except Exception as e:
            print_error(f"Validation process failed: {str(e)}")
            logger.error(traceback.format_exc())
            self.results['tests_failed'].append(f"IFC validation: {str(e)}")
            return False
    
    def test_file_characteristics(self):
        """Test file size and complexity"""
        print_header("TEST 6: File Characteristics")
        
        if not self.results['ifc_files_generated']:
            print_warning("No IFC files to analyze")
            return False
        
        try:
            import ifcopenshell
            
            for ifc_path in self.results['ifc_files_generated']:
                print_info(f"Analyzing: {Path(ifc_path).name}")
                
                # File size
                file_size = Path(ifc_path).stat().st_size
                file_size_mb = file_size / (1024 * 1024)
                print_info(f"  File size: {file_size:,} bytes ({file_size_mb:.2f} MB)")
                
                if file_size_mb > 10:
                    print_warning(f"  Large file size may cause rendering issues")
                    self.results['warnings'].append(f"{Path(ifc_path).name}: Large file size")
                elif file_size_mb > 50:
                    print_error(f"  Very large file - will likely fail to render in browser")
                    self.results['warnings'].append(f"{Path(ifc_path).name}: Very large file")
                else:
                    print_success(f"  File size is acceptable for web rendering")
                
                # Entity count
                ifc_file = ifcopenshell.open(ifc_path)
                entity_count = len(list(ifc_file))
                print_info(f"  Entity count: {entity_count:,}")
                
                if entity_count > 10000:
                    print_warning(f"  High entity count may slow rendering")
                    self.results['warnings'].append(f"{Path(ifc_path).name}: High entity count")
                
                # Geometry complexity
                products_with_geometry = [
                    e for e in ifc_file.by_type("IfcProduct")
                    if e.Representation is not None
                ]
                print_info(f"  Products with geometry: {len(products_with_geometry)}")
            
            self.results['tests_passed'].append("File characteristics analysis")
            return True
            
        except Exception as e:
            print_error(f"File analysis failed: {str(e)}")
            logger.error(traceback.format_exc())
            self.results['tests_failed'].append(f"File characteristics: {str(e)}")
            return False
    
    def test_coordinate_system(self):
        """Test coordinate system and placement"""
        print_header("TEST 7: Coordinate System Analysis")
        
        if not self.results['ifc_files_generated']:
            print_warning("No IFC files to analyze")
            return False
        
        try:
            import ifcopenshell
            import numpy as np
            
            for ifc_path in self.results['ifc_files_generated']:
                print_info(f"Analyzing coordinates: {Path(ifc_path).name}")
                
                ifc_file = ifcopenshell.open(ifc_path)
                
                # Check all placements
                placements = ifc_file.by_type("IfcLocalPlacement")
                
                if not placements:
                    print_warning("  No placements found")
                    continue
                
                # Analyze coordinate ranges
                all_coords = []
                for placement in placements:
                    if placement.RelativePlacement:
                        axis = placement.RelativePlacement
                        if hasattr(axis, 'Location') and axis.Location:
                            coords = axis.Location.Coordinates
                            all_coords.append(coords)
                
                if all_coords:
                    coords_array = np.array(all_coords)
                    min_coords = coords_array.min(axis=0)
                    max_coords = coords_array.max(axis=0)
                    center = coords_array.mean(axis=0)
                    
                    print_info(f"  Coordinate ranges:")
                    print_info(f"    X: {min_coords[0]:.2f} to {max_coords[0]:.2f}")
                    print_info(f"    Y: {min_coords[1]:.2f} to {max_coords[1]:.2f}")
                    print_info(f"    Z: {min_coords[2]:.2f} to {max_coords[2]:.2f}")
                    print_info(f"  Model center: ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})")
                    
                    # Check if model is far from origin
                    distance_from_origin = np.linalg.norm(center)
                    if distance_from_origin > 1000:
                        print_error(f"  Model is very far from origin ({distance_from_origin:.2f}m)")
                        print_error(f"  This WILL cause rendering issues!")
                        self.results['warnings'].append(
                            f"{Path(ifc_path).name}: Model far from origin - CRITICAL"
                        )
                    elif distance_from_origin > 100:
                        print_warning(f"  Model is somewhat far from origin ({distance_from_origin:.2f}m)")
                        self.results['warnings'].append(
                            f"{Path(ifc_path).name}: Model offset from origin"
                        )
                    else:
                        print_success(f"  Model is reasonably centered")
                else:
                    print_warning("  No coordinate data found")
            
            self.results['tests_passed'].append("Coordinate system analysis")
            return True
            
        except Exception as e:
            print_error(f"Coordinate analysis failed: {str(e)}")
            logger.error(traceback.format_exc())
            self.results['tests_failed'].append(f"Coordinate system: {str(e)}")
            return False
    
    def test_backend_api(self):
        """Test backend API if it's running"""
        print_header("TEST 8: Backend API Testing")
        
        try:
            import requests
            
            # Test if backend is running
            backend_url = "http://localhost:8000"
            
            print_info(f"Testing connection to {backend_url}...")
            
            try:
                response = requests.get(f"{backend_url}/health", timeout=2)
                if response.status_code == 200:
                    print_success("Backend is running")
                    
                    # Test other endpoints if available
                    print_info("Testing /docs endpoint...")
                    docs_response = requests.get(f"{backend_url}/docs", timeout=2)
                    if docs_response.status_code == 200:
                        print_success("API documentation accessible")
                    
                    self.results['tests_passed'].append("Backend API connection")
                else:
                    print_warning(f"Backend responded with status {response.status_code}")
                    
            except requests.exceptions.ConnectionError:
                print_warning("Backend is not running")
                print_info("To start backend: cd backend && uvicorn main:app --reload")
                self.results['warnings'].append("Backend not running")
            except requests.exceptions.Timeout:
                print_warning("Backend connection timeout")
                self.results['warnings'].append("Backend timeout")
                
        except ImportError:
            print_warning("requests library not available for API testing")
        except Exception as e:
            print_error(f"API testing failed: {str(e)}")
            logger.error(traceback.format_exc())
    
    def generate_report(self):
        """Generate comprehensive diagnostic report"""
        print_header("DIAGNOSTIC REPORT")
        
        # Summary
        total_tests = len(self.results['tests_passed']) + len(self.results['tests_failed'])
        pass_rate = (len(self.results['tests_passed']) / total_tests * 100) if total_tests > 0 else 0
        
        print_info(f"Total Tests: {total_tests}")
        print_success(f"Passed: {len(self.results['tests_passed'])}")
        print_error(f"Failed: {len(self.results['tests_failed'])}")
        print_warning(f"Warnings: {len(self.results['warnings'])}")
        print_info(f"Pass Rate: {pass_rate:.1f}%")
        
        # Detailed results
        if self.results['tests_passed']:
            print_info("\n✓ Tests Passed:")
            for test in self.results['tests_passed']:
                print_success(f"  - {test}")
        
        if self.results['tests_failed']:
            print_info("\n✗ Tests Failed:")
            for test in self.results['tests_failed']:
                print_error(f"  - {test}")
        
        if self.results['warnings']:
            print_info("\n⚠ Warnings:")
            for warning in self.results['warnings']:
                print_warning(f"  - {warning}")
        
        # Generated files
        if self.results['ifc_files_generated']:
            print_info("\n📁 Generated IFC Files:")
            for file_path in self.results['ifc_files_generated']:
                print_info(f"  - {file_path}")
        
        # Save JSON report
        report_file = self.output_dir / f"diagnostic_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print_info(f"\n📊 JSON report saved to: {report_file}")
        print_info(f"📋 Log file saved to: {log_filename}")
        
        # Recommendations
        print_header("RECOMMENDATIONS")
        
        if any("Model far from origin" in w for w in self.results['warnings']):
            print_warning("CRITICAL: Your models are far from origin!")
            print_info("  → This is the most common cause of rendering failures")
            print_info("  → Solution: Center your geometry around (0, 0, 0)")
            print_info("  → Use IfcLocalPlacement with origin at or near (0, 0, 0)")
        
        if any("Large file" in w for w in self.results['warnings']):
            print_warning("File size may cause issues:")
            print_info("  → Optimize geometry complexity")
            print_info("  → Remove unnecessary details")
            print_info("  → Consider level-of-detail (LOD) strategies")
        
        if "Backend not running" in str(self.results['warnings']):
            print_info("Backend is not running:")
            print_info("  → cd backend")
            print_info("  → uvicorn main:app --reload")
        
        if len(self.results['tests_failed']) == 0:
            print_success("\n🎉 All tests passed! Your IFC generation is working correctly.")
        else:
            print_error("\n❌ Some tests failed. Check the log file for details.")
        
        print_info(f"\n{'='*80}")
        print_info("For viewer issues, check:")
        print_info("  1. Browser console (F12) for JavaScript errors")
        print_info("  2. Network tab for failed WASM file loads")
        print_info("  3. IFC file is accessible from the frontend")
        print_info("  4. WASM path is correctly configured in viewer")
        print_info(f"{'='*80}\n")


def main():
    """Main entry point"""
    print_info("Starting IFC Diagnostic Suite...")
    print_info(f"Output directory: diagnostic_output/")
    print_info(f"Log file: {log_filename}\n")
    
    tester = IFCDiagnosticTester()
    
    try:
        tester.run_all_tests()
    except KeyboardInterrupt:
        print_warning("\n\nTests interrupted by user")
    except Exception as e:
        print_error(f"\n\nUnexpected error: {str(e)}")
        logger.error(traceback.format_exc())
    finally:
        print_info(f"\nDiagnostic complete. Check {log_filename} for details.")


if __name__ == "__main__":
    # Import multiprocessing at module level for geometry iterator
    try:
        import multiprocessing
    except ImportError:
        multiprocessing = None
        logger.warning("multiprocessing not available - geometry iterator may be slower")
    
    main()