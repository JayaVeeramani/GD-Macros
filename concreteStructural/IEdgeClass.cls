VERSION 1.0 CLASS
BEGIN
  MultiUse = -1  'True
END
Attribute VB_Name = "IEdgeClass"
Attribute VB_GlobalNameSpace = False
Attribute VB_Creatable = False
Attribute VB_PredeclaredId = False
Attribute VB_Exposed = False
Private cEdge As SldWorks.Edge
Private cComp As SldWorks.Component2
Private cPoint As Double
Private cLength As Double
Private cIsVertical As Boolean

Private VerEdgeList As IArrListObject

Property Get GetEdge() As SldWorks.Edge
    Set GetEdge = cEdge
End Property

Property Get IsVertical() As Boolean
    IsVertical = cIsVertical
End Property

Property Get Point() As Double

    Point = cPoint
    
End Property

Property Get Length() As Double

    Length = cLength
    
End Property


Sub Initialize(swEdge As SldWorks.Edge, IsVertical As Boolean, Point As Double, Length As Double)

    Set cEdge = swEdge
    cIsVertical = IsVertical
    cPoint = Point
    cLength = Length

End Sub

