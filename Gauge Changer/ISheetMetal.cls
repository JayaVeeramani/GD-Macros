VERSION 1.0 CLASS
BEGIN
  MultiUse = -1  'True
END
Attribute VB_Name = "ISheetMetal"
Attribute VB_GlobalNameSpace = False
Attribute VB_Creatable = False
Attribute VB_PredeclaredId = False
Attribute VB_Exposed = False
Private cPart As PartDoc
Private cBaseFlangeDefinition As SldWorks.BaseFlangeFeatureData
Private cBaseFlangeFeat As SldWorks.Feature
Private cGauge As String

Property Get Thickness() As Double
    
    Select Case cGauge

        Case "12GA"
            Thickness = 0.1084

        Case "14GA"
            Thickness = 0.0785
    
        Case "16GA"
            Thickness = 0.0635
    
        Case "18GA"
            Thickness = 0.0516
    
        Case Else
            Thickness = 0

    End Select

End Property

Property Get Gauge() As String

    Gauge = cGauge
    
End Property

Property Get GetPartDoc() As PartDoc

    Set GetPartDoc = cPart
    
End Property

Sub Initialize(Part As PartDoc)

    Set cPart = Part
    Call GetBaseFlangeData
    cGauge = cBaseFlangeDefinition.ThicknessTableName
    
End Sub

Public Sub GetBaseFlangeData()

    Dim swFeat As IFeature
    Set swFeat = cPart.GetModelDocObject.FirstFeature
    
    Do While Not swFeat Is Nothing
    
        Select Case swFeat.GetTypeName

            Case "SMBaseFlange"
            
                Set cBaseFlangeFeat = swFeat
                Set cBaseFlangeDefinition = swFeat.GetDefinition
                
'                Select Case cBaseFlangeDefinition.ThicknessTableName
'
'                    Case "12GA"
'                        GetThickness = 0.1084
'
'                    Case "14GA"
'                         GetThickness = 0.0785
'
'                    Case "16GA"
'                        GetThickness = 0.0635
'
'                    Case "18GA"
'                         GetThickness = 0.0516
'
'                    Case Else
'                        GetThickness = 0
'
'                End Select
                Exit Sub
                
            'Case "SheetMetal"
                'Process_SheetMetal swApp, swModel, swFeat

        End Select
        
        Set swFeat = swFeat.GetNextFeature

    Loop

End Sub

Public Function UpdateThickness(GaugeVal As String) As Boolean
      
    Dim vThicknesses As Variant
    vThicknesses = cBaseFlangeDefinition.GetTableThicknesses()
                                            
    Dim ThkCount As Integer
    Dim ThkExists As Boolean
    For ThkCount = LBound(vThicknesses) To UBound(vThicknesses)
                                               
        If vThicknesses(ThkCount) = GaugeVal Then
            ThkExists = True
            Exit For
        End If
                                        
    Next ThkCount
    
    If ThkExists Then
    
        cBaseFlangeDefinition.ThicknessTableName = GaugeVal
        cBaseFlangeFeat.ModifyDefinition cBaseFlangeDefinition, cPart.GetModelDocObject, Nothing
        
    End If
    
End Function
