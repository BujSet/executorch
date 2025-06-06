# automatically generated by the FlatBuffers compiler, do not modify

# namespace: tflite

import flatbuffers
from flatbuffers.compat import import_numpy

np = import_numpy()


class SubGraph(object):
    __slots__ = ["_tab"]

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = SubGraph()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsSubGraph(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)

    @classmethod
    def SubGraphBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(
            buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed
        )

    # SubGraph
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # SubGraph
    def Tensors(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            x = self._tab.Vector(o)
            x += flatbuffers.number_types.UOffsetTFlags.py_type(j) * 4
            x = self._tab.Indirect(x)
            from .Tensor import Tensor

            obj = Tensor()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # SubGraph
    def TensorsLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # SubGraph
    def TensorsIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        return o == 0

    # SubGraph
    def Inputs(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(
                flatbuffers.number_types.Int32Flags,
                a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 4),
            )
        return 0

    # SubGraph
    def InputsAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Int32Flags, o)
        return 0

    # SubGraph
    def InputsLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # SubGraph
    def InputsIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        return o == 0

    # SubGraph
    def Outputs(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(
                flatbuffers.number_types.Int32Flags,
                a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 4),
            )
        return 0

    # SubGraph
    def OutputsAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Int32Flags, o)
        return 0

    # SubGraph
    def OutputsLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # SubGraph
    def OutputsIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        return o == 0

    # SubGraph
    def Operators(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            x = self._tab.Vector(o)
            x += flatbuffers.number_types.UOffsetTFlags.py_type(j) * 4
            x = self._tab.Indirect(x)
            from .Operator import Operator

            obj = Operator()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # SubGraph
    def OperatorsLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # SubGraph
    def OperatorsIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        return o == 0

    # SubGraph
    def Name(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(12))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None


def SubGraphStart(builder):
    builder.StartObject(5)


def Start(builder):
    SubGraphStart(builder)


def SubGraphAddTensors(builder, tensors):
    builder.PrependUOffsetTRelativeSlot(
        0, flatbuffers.number_types.UOffsetTFlags.py_type(tensors), 0
    )


def AddTensors(builder, tensors):
    SubGraphAddTensors(builder, tensors)


def SubGraphStartTensorsVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)


def StartTensorsVector(builder, numElems: int) -> int:
    return SubGraphStartTensorsVector(builder, numElems)


def SubGraphAddInputs(builder, inputs):
    builder.PrependUOffsetTRelativeSlot(
        1, flatbuffers.number_types.UOffsetTFlags.py_type(inputs), 0
    )


def AddInputs(builder, inputs):
    SubGraphAddInputs(builder, inputs)


def SubGraphStartInputsVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)


def StartInputsVector(builder, numElems: int) -> int:
    return SubGraphStartInputsVector(builder, numElems)


def SubGraphAddOutputs(builder, outputs):
    builder.PrependUOffsetTRelativeSlot(
        2, flatbuffers.number_types.UOffsetTFlags.py_type(outputs), 0
    )


def AddOutputs(builder, outputs):
    SubGraphAddOutputs(builder, outputs)


def SubGraphStartOutputsVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)


def StartOutputsVector(builder, numElems: int) -> int:
    return SubGraphStartOutputsVector(builder, numElems)


def SubGraphAddOperators(builder, operators):
    builder.PrependUOffsetTRelativeSlot(
        3, flatbuffers.number_types.UOffsetTFlags.py_type(operators), 0
    )


def AddOperators(builder, operators):
    SubGraphAddOperators(builder, operators)


def SubGraphStartOperatorsVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)


def StartOperatorsVector(builder, numElems: int) -> int:
    return SubGraphStartOperatorsVector(builder, numElems)


def SubGraphAddName(builder, name):
    builder.PrependUOffsetTRelativeSlot(
        4, flatbuffers.number_types.UOffsetTFlags.py_type(name), 0
    )


def AddName(builder, name):
    SubGraphAddName(builder, name)


def SubGraphEnd(builder):
    return builder.EndObject()


def End(builder):
    return SubGraphEnd(builder)
