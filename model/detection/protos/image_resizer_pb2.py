# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: model/detection/protos/image_resizer.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database

# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='model/detection/protos/image_resizer.proto',
  package='model.detection.protos',
  serialized_pb=_b('\n*model/detection/protos/image_resizer.proto\x12\x16model.detection.protos\"\xc4\x01\n\x0cImageResizer\x12S\n\x19keep_aspect_ratio_resizer\x18\x01 \x01(\x0b\x32..model.detection.protos.KeepAspectRatioResizerH\x00\x12H\n\x13\x66ixed_shape_resizer\x18\x02 \x01(\x0b\x32).model.detection.protos.FixedShapeResizerH\x00\x42\x15\n\x13image_resizer_oneof\"Q\n\x16KeepAspectRatioResizer\x12\x1a\n\rmin_dimension\x18\x01 \x01(\x05:\x03\x36\x30\x30\x12\x1b\n\rmax_dimension\x18\x02 \x01(\x05:\x04\x31\x30\x32\x34\"<\n\x11\x46ixedShapeResizer\x12\x13\n\x06height\x18\x01 \x01(\x05:\x03\x33\x30\x30\x12\x12\n\x05width\x18\x02 \x01(\x05:\x03\x33\x30\x30')
)
_sym_db.RegisterFileDescriptor(DESCRIPTOR)




_IMAGERESIZER = _descriptor.Descriptor(
  name='ImageResizer',
  full_name='model.detection.protos.ImageResizer',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='keep_aspect_ratio_resizer', full_name='model.detection.protos.ImageResizer.keep_aspect_ratio_resizer', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='fixed_shape_resizer', full_name='model.detection.protos.ImageResizer.fixed_shape_resizer', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
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
    _descriptor.OneofDescriptor(
      name='image_resizer_oneof', full_name='model.detection.protos.ImageResizer.image_resizer_oneof',
      index=0, containing_type=None, fields=[]),
  ],
  serialized_start=71,
  serialized_end=267,
)


_KEEPASPECTRATIORESIZER = _descriptor.Descriptor(
  name='KeepAspectRatioResizer',
  full_name='model.detection.protos.KeepAspectRatioResizer',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='min_dimension', full_name='model.detection.protos.KeepAspectRatioResizer.min_dimension', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=600,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='max_dimension', full_name='model.detection.protos.KeepAspectRatioResizer.max_dimension', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=1024,
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
  serialized_start=269,
  serialized_end=350,
)


_FIXEDSHAPERESIZER = _descriptor.Descriptor(
  name='FixedShapeResizer',
  full_name='model.detection.protos.FixedShapeResizer',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='height', full_name='model.detection.protos.FixedShapeResizer.height', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=300,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='width', full_name='model.detection.protos.FixedShapeResizer.width', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=300,
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
  serialized_start=352,
  serialized_end=412,
)

_IMAGERESIZER.fields_by_name['keep_aspect_ratio_resizer'].message_type = _KEEPASPECTRATIORESIZER
_IMAGERESIZER.fields_by_name['fixed_shape_resizer'].message_type = _FIXEDSHAPERESIZER
_IMAGERESIZER.oneofs_by_name['image_resizer_oneof'].fields.append(
  _IMAGERESIZER.fields_by_name['keep_aspect_ratio_resizer'])
_IMAGERESIZER.fields_by_name['keep_aspect_ratio_resizer'].containing_oneof = _IMAGERESIZER.oneofs_by_name['image_resizer_oneof']
_IMAGERESIZER.oneofs_by_name['image_resizer_oneof'].fields.append(
  _IMAGERESIZER.fields_by_name['fixed_shape_resizer'])
_IMAGERESIZER.fields_by_name['fixed_shape_resizer'].containing_oneof = _IMAGERESIZER.oneofs_by_name['image_resizer_oneof']
DESCRIPTOR.message_types_by_name['ImageResizer'] = _IMAGERESIZER
DESCRIPTOR.message_types_by_name['KeepAspectRatioResizer'] = _KEEPASPECTRATIORESIZER
DESCRIPTOR.message_types_by_name['FixedShapeResizer'] = _FIXEDSHAPERESIZER

ImageResizer = _reflection.GeneratedProtocolMessageType('ImageResizer', (_message.Message,), dict(
  DESCRIPTOR = _IMAGERESIZER,
  __module__ = 'model.detection.protos.image_resizer_pb2'
  # @@protoc_insertion_point(class_scope:model.detection.protos.ImageResizer)
  ))
_sym_db.RegisterMessage(ImageResizer)

KeepAspectRatioResizer = _reflection.GeneratedProtocolMessageType('KeepAspectRatioResizer', (_message.Message,), dict(
  DESCRIPTOR = _KEEPASPECTRATIORESIZER,
  __module__ = 'model.detection.protos.image_resizer_pb2'
  # @@protoc_insertion_point(class_scope:model.detection.protos.KeepAspectRatioResizer)
  ))
_sym_db.RegisterMessage(KeepAspectRatioResizer)

FixedShapeResizer = _reflection.GeneratedProtocolMessageType('FixedShapeResizer', (_message.Message,), dict(
  DESCRIPTOR = _FIXEDSHAPERESIZER,
  __module__ = 'model.detection.protos.image_resizer_pb2'
  # @@protoc_insertion_point(class_scope:model.detection.protos.FixedShapeResizer)
  ))
_sym_db.RegisterMessage(FixedShapeResizer)


# @@protoc_insertion_point(module_scope)
