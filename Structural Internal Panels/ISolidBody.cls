VERSION 1.0 CLASS
BEGIN
  MultiUse = -1  'True
END
Attribute VB_Name = "ISolidBody"
Attribute VB_GlobalNameSpace = False
Attribute VB_Creatable = False
Attribute VB_PredeclaredId = False
Attribute VB_Exposed = False
Private cComp As SldWorks.Component2
Private cBody As SldWorks.Body2
Private cXMin As Double
Private cYMin As Double
Private cZMin As Double

Private cXMax As Double
Private cYMax As Double
Private cZMax As Double

Private cModelMinPoint As Variant
Private cModelMaxPoint As Variant


Property Get GetBody() As SldWorks.Body2

    Set GetBody = cBody
    
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

Property Get GetComponent() As SldWorks.Component2
    Set GetComponent = cComp
End Property


Public Sub Initialize(body As SldWorks.Body2, swComp As SldWorks.Component2, swView As SldWorks.View)

    Set cBody = body
    Set cComp = swComp
    
    Dim vBodyBox As Variant
    vBodyBox = body.GetBodyBox
    
    Dim vPoint(2) As Double
    vPoint(0) = vBodyBox(0)
    vPoint(1) = vBodyBox(1)
    vPoint(2) = vBodyBox(2)
    
    
    cModelMinPoint = vPoint
    
    vPoint(0) = vBodyBox(3)
    vPoint(1) = vBodyBox(4)
    vPoint(2) = vBodyBox(5)
    
    cModelMaxPoint = vPoint
    
    Dim MinPoint As Variant
    Dim MaxPoint As Variant
    MinPoint = GetComponentPointInSheetSpace(swComp, cModelMinPoint, swView)
    MaxPoint = GetComponentPointInSheetSpace(swComp, cModelMaxPoint, swView)
    
    Call StrucutralElevationInsulation.GetMaxMinPoint(MinPoint(0), MaxPoint(0), cXMin, cXMax)
    Call StrucutralElevationInsulation.GetMaxMinPoint(MinPoint(1), MaxPoint(1), cYMin, cYMax)
    Call StrucutralElevationInsulation.GetMaxMinPoint(MinPoint(2), MaxPoint(2), cZMin, cZMax)
    
End Sub
