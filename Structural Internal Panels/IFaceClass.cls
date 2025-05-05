VERSION 1.0 CLASS
BEGIN
  MultiUse = -1  'True
END
Attribute VB_Name = "IFaceClass"
Attribute VB_GlobalNameSpace = False
Attribute VB_Creatable = False
Attribute VB_PredeclaredId = False
Attribute VB_Exposed = False
Private cFace As SldWorks.Face2
Private cComp As SldWorks.Component2
Private cXMin As Double
Private cYMin As Double
Private cZMin As Double

Private cXMax As Double
Private cYMax As Double
Private cZMax As Double

Property Get GetFace() As SldWorks.Face2
    Set GetFace = cFace
End Property

Property Get GetComponent() As SldWorks.Component2
    Set GetComponent = cComp
End Property

Property Get xMin() As Double
    xMin = cXMin
End Property

Property Get yMin() As Double
    yMin = cYMin
End Property

Property Get zMin() As Double
    zMin = cZMin
End Property

Property Get xMax() As Double
    xMax = cXMax
End Property

Property Get yMax() As Double
    yMax = cYMax
End Property

Property Get zMax() As Double
    zMax = cZMax
End Property

Sub Initialize(swFace As SldWorks.Face2, swView As SldWorks.View, swComp As SldWorks.Component2)

    Set cFace = swFace
    Set cComp = swComp
    
    Dim vFaceBox As Variant
    vFaceBox = swFace.GetBox
    
    Dim vFaceMinPoint(2) As Double
    Dim vFaceMaxPoint(2) As Double
    
    vFaceMinPoint(0) = vFaceBox(0)
    vFaceMinPoint(1) = vFaceBox(1)
    vFaceMinPoint(2) = vFaceBox(2)
            
    vFaceMaxPoint(0) = vFaceBox(3)
    vFaceMaxPoint(1) = vFaceBox(4)
    vFaceMaxPoint(2) = vFaceBox(5)

            
    MinPoint = GetComponentPointInSheetSpace(swComp, vFaceMinPoint, swView)
    MaxPoint = GetComponentPointInSheetSpace(swComp, vFaceMaxPoint, swView)
    
    Call GetMaxMinPoint(MinPoint(0), MaxPoint(0), cXMin, cXMax)
    Call GetMaxMinPoint(MinPoint(1), MaxPoint(1), cYMin, cYMax)
    Call GetMaxMinPoint(MinPoint(2), MaxPoint(2), cZMin, cZMax)
    
End Sub


