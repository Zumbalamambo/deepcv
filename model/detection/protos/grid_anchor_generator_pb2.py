# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: model/detection/protos/grid_anchor_generator.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='model/detection/protos/grid_anchor_generator.proto',
  package='model.detection.protos',
  serialized_pb=_b('\n2model/detection/protos/grid_anchor_generator.proto\x12\x16model.detection.protos\"\xcd\x01\n\x13GridAnchorGenerator\x12\x13\n\x06height\x18\x01 \x01(\x05:\x03\x32\x35\x36\x12\x12\n\x05width\x18\x02 \x01(\x05:\x03\x32\x35\x36\x12\x19\n\rheight_stride\x18\x03 \x01(\x05:\x02\x31\x36\x12\x18\n\x0cwidth_stride\x18\x04 \x01(\x05:\x02\x31\x36\x12\x18\n\rheight_offset\x18\x05 \x01(\x05:\x01\x30\x12\x17\n\x0cwidth_offset\x18\x06 \x01(\x05:\x01\x30\x12\x0e\n\x06scales\x18\x07 \x03(\x02\x12\x15\n\raspect_ratios\x18\x08 \x03(\x02')
)
_sym_db.RegisterFileDescriptor(DESCRIPTOR)




_GRIDANCHORGENERATOR = _descriptor.Descriptor(
  name='GridAnchorGenerator',
  full_name='model.detection.protos.GridAnchorGenerator',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='height', full_name='model.detection.protos.GridAnchorGenerator.height', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=256,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='width', full_name='model.detection.protos.GridAnchorGenerator.width', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=256,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='height_stride', full_name='model.detection.protos.GridAnchorGenerator.height_stride', index=2,
      number=3, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=16,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='width_stride', full_name='model.detection.protos.GridAnchorGenerator.width_stride', index=3,
      number=4, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=16,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='height_offset', full_name='model.detection.protos.GridAnchorGenerator.height_offset', index=4,
      number=5, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='width_offset', full_name='model.detection.protos.GridAnchorGenerator.width_offset', index=5,
      number=6, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='scales', full_name='model.detection.protos.GridAnchorGenerator.scales', index=6,
      number=7, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='aspect_ratios', full_name='model.detection.protos.GridAnchorGenerator.aspect_ratios', index=7,
      number=8, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=79,
  serialized_end=284,
)

DESCRIPTOR.message_types_by_name['GridAnchorGenerator'] = _GRIDANCHORGENERATOR

GridAnchorGenerator = _reflection.GeneratedProtocolMessageType('GridAnchorGenerator', (_message.Message,), dict(
  DESCRIPTOR = _GRIDANCHORGENERATOR,
  __module__ = 'model.detection.protos.grid_anchor_generator_pb2'
  # @@protoc_insertion_point(class_scope:model.detection.protos.GridAnchorGenerator)
  ))
_sym_db.RegisterMessage(GridAnchorGenerator)


# @@protoc_insertion_point(module_scope)
