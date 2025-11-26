VERSION 1.0 CLASS
BEGIN
  MultiUse = -1  'True
END
Attribute VB_Name = "ISketchConvertInput"
Attribute VB_GlobalNameSpace = False
Attribute VB_Creatable = False
Attribute VB_PredeclaredId = False
Attribute VB_Exposed = False
Public vCompArr As Variant
Public swSketchPoint As SldWorks.sketchPoint
Public swVertex As SldWorks.Vertex
Public PointComp As SldWorks.Component2

Public ProjectNo As String



