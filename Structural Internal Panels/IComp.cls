VERSION 1.0 CLASS
BEGIN
  MultiUse = -1  'True
END
Attribute VB_Name = "IComp"
Attribute VB_GlobalNameSpace = False
Attribute VB_Creatable = False
Attribute VB_PredeclaredId = False
Attribute VB_Exposed = False
Private cComp As SldWorks.Component2
Private cXMin As Double
Private cYMin As Double
Private cZMin As Double

Private cXMax As Double
Private cYMax As Double
Private cZMax As Double

'Private cViewXMin As Double
'Private cViewYMin As Double
'Private cViewZMin As Double
'
'Private cViewXMax As Double
'Private cViewYMax As Double
'Private cViewZMax As Double

Private cIsMakeUp As Boolean

Private cModelMinPoint As Variant
Private cModelMaxPoint As Variant
Private cisTop As Boolean
Private cisBottom As Boolean


Property Get GetMinPointInModel() As Variant
    GetMinPointInModel = cModelMinPoint
End Property

Property Get GetMaxPointInModel() As Variant
    GetMaxPointInModel = cModelMaxPoint
End Property

Property Get LargestFace() As SldWorks.Face2
    Set LargestFace = cFace
End Property

Property Get IsMakeUpPanel() As Boolean
    IsMakeUpPanel = cIsMakeUp
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


'Property Get xViewMin() As Double
'    xViewMin = cViewXMin
'End Property
'
'Property Get yViewMin() As Double
'    yViewMin = cViewYMin
'End Property
'
'Property Get zViewMin() As Double
'    zViewMin = cViewZMin
'End Property
'
'Property Get xViewMax() As Double
'    xViewMax = cViewXMax
'End Property
'
'Property Get yViewMax() As Double
'    yViewMax = cViewYMax
'End Property
'
'Property Get zViewMax() As Double
'    zViewMax = cViewZMax
'End Property


Property Get GetComponent() As SldWorks.Component2
    Set GetComponent = cComp
End Property

Property Get GetPathName() As String

    GetPathName = cComp.GetModelDoc2.GetPathName()
    
End Property

Property Let IsTop(Val As Boolean)

    cisTop = Val
    
End Property

Property Get IsTop() As Boolean

    IsTop = cisTop
    
End Property

Property Let IsBottom(Val As Boolean)

    cisBottom = Val
    
End Property

Property Get IsBottom() As Boolean

    IsBottom = cisBottom
    
End Property



Sub Initialize(swComp As IComponent2, Optional MinPoint As Variant, Optional MaxPoint As Variant, _
            Optional ModelMinPoint As Variant, Optional ModelMaxPoint As Variant)

    Set cComp = swComp
    
    Call GetMaxMinPoint(MinPoint(0), MaxPoint(0), cXMin, cXMax)
    Call GetMaxMinPoint(MinPoint(1), MaxPoint(1), cYMin, cYMax)
    Call GetMaxMinPoint(MinPoint(2), MaxPoint(2), cZMin, cZMax)

    cModelMinPoint = ModelMinPoint
    cModelMaxPoint = ModelMaxPoint
    
    cIsMakeUp = False
    If InStr(GetCustomProperty("REMARKS"), "MAKE") > 0 Then
        
        cIsMakeUp = True
        
    End If
    
    cisTop = True
    cisBottom = False

End Sub

Function GetCustomProperty(PropName As String) As String

    Dim swCompModel As SldWorks.ModelDoc2
    Set swCompModel = cComp.GetModelDoc2
 
    Dim swCompProp As SldWorks.CustomPropertyManager
    Set swCompProp = swCompModel.Extension.CustomPropertyManager("")
            
    Dim PropVal As String
    Dim ValOut As String
    Dim wasResolved As Boolean
    swCompProp.Get5 PropName, False, ValOut, PropVal, wasResolved
    
    GetCustomProperty = PropVal
    
End Function


'Private Sub GetMaxMinPoint(LowerBoundPoint As Variant, HigherBoundPoint As Variant, _
'            ByRef OutLower As Double, ByRef OutHigher As Double)
'
'    If LowerBoundPoint < HigherBoundPoint Then
'
'        OutLower = LowerBoundPoint
'        OutHigher = HigherBoundPoint
'
'    Else
'
'        OutLower = HigherBoundPoint
'        OutHigher = LowerBoundPoint
'
'    End If
'
'End Sub

Sub UpdateSheetMaxMinDimensions(swView As SldWorks.View, MinPoint As Variant, MaxPoint As Variant)

    Call GetMaxMinPoint(MinPoint(0), MaxPoint(0), cXMin, cXMax)
    Call GetMaxMinPoint(MinPoint(1), MaxPoint(1), cYMin, cYMax)
    Call GetMaxMinPoint(MinPoint(2), MaxPoint(2), cZMin, cZMax)
    
End Sub







