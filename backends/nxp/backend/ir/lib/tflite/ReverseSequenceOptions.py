# automatically generated by the FlatBuffers compiler, do not modify

# namespace: tflite

import flatbuffers
from flatbuffers.compat import import_numpy

np = import_numpy()


class ReverseSequenceOptions(object):
    __slots__ = ["_tab"]

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = ReverseSequenceOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsReverseSequenceOptions(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)

    @classmethod
    def ReverseSequenceOptionsBufferHasIdentifier(
        cls, buf, offset, size_prefixed=False
    ):
        return flatbuffers.util.BufferHasIdentifier(
            buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed
        )

    # ReverseSequenceOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # ReverseSequenceOptions
    def SeqDim(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # ReverseSequenceOptions
    def BatchDim(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0


def ReverseSequenceOptionsStart(builder):
    builder.StartObject(2)


def Start(builder):
    ReverseSequenceOptionsStart(builder)


def ReverseSequenceOptionsAddSeqDim(builder, seqDim):
    builder.PrependInt32Slot(0, seqDim, 0)


def AddSeqDim(builder, seqDim):
    ReverseSequenceOptionsAddSeqDim(builder, seqDim)


def ReverseSequenceOptionsAddBatchDim(builder, batchDim):
    builder.PrependInt32Slot(1, batchDim, 0)


def AddBatchDim(builder, batchDim):
    ReverseSequenceOptionsAddBatchDim(builder, batchDim)


def ReverseSequenceOptionsEnd(builder):
    return builder.EndObject()


def End(builder):
    return ReverseSequenceOptionsEnd(builder)
