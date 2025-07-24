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
Private cxMin As Double
Private cyMin As Double
Private cZMin As Double

Private cxMax As Double
Private cyMax As Double
Private cZMax As Double

Private cxOrigin As Double
Private cyOrigin As Double
Private czOrigin As Double

Private cView As SldWorks.View
Private cModelMinPoint(2) As Double
Private cModelMaxPoint(2) As Double

Private cBodiesList As IArrListObject
Private Part As SldWorks.PartDoc

Private IsBodiesDataFound As Boolean

Private orgViewPosition As Variant
Private orgScaleDecimal As Double

Property Get GetMinPointInModel() As Variant
    GetMinPointInModel = cModelMinPoint
End Property

Property Get GetMaxPointInModel() As Variant
    GetMaxPointInModel = cModelMaxPoint
End Property

Property Get Name() As String

    Name = cComp.Name2
    
End Property

Property Get xOrgin() As Double
 
    xOrgin = cxOrigin
    
End Property

Property Get yOrgin() As Double
 
    yOrgin = cyOrigin
    
End Property

Property Get zOrgin() As Double
 
    zOrgin = czOrigin
    
End Property


Property Get xMin() As Double
 
    xMin = cxMin
    
End Property

Property Get yMin() As Double

    yMin = cyMin
    
End Property

Property Get zMin() As Double

    zMin = cZMin
    
End Property

Property Get xMax() As Double

    xMax = cxMax
    
End Property

Property Get yMax() As Double

    yMax = cyMax
    
End Property

Property Get zMax() As Double

    zMax = cZMax
    
End Property


Property Get GetComponent() As SldWorks.Component2

    Set GetComponent = cComp
    
End Property

Property Get GetPathName() As String

    GetPathName = cComp.GetModelDoc2.GetPathName()
    
End Property

Sub Initialize(swComp As IComponent2, swView As SldWorks.View)

    Set cComp = swComp
    Set swCompModel = ResolveAndGetModelDoc(swComp)
    
    Set cBodiesList = New IArrListObject
    
    IsBodiesDataFound = False
        
    If swCompModel.GetType = swDocumentTypes_e.swDocPART Then

        Set Part = swCompModel
        
        Dim vCompBounds As Variant
        vCompBounds = swCompModel.GetPartBox(True)
        
        
        cModelMinPoint(0) = vCompBounds(0)
        cModelMinPoint(1) = vCompBounds(1)
        cModelMinPoint(2) = vCompBounds(2)
        
        cModelMaxPoint(0) = vCompBounds(3)
        cModelMaxPoint(1) = vCompBounds(4)
        cModelMaxPoint(2) = vCompBounds(5)
        
        Set cView = swView
        
        orgViewPosition = cView.Position
        orgScaleDecimal = cView.ScaleDecimal
        
        Call UpdateSheetMaxMinDimensions(cView)
        Call GetOriginDimensions(swView)
        
    End If

End Sub

 Function GetBodiesList(swView As SldWorks.View, swNormalVector As SldWorks.MathVector) As IArrListObject
    
    If cBodiesList.Count = 0 Then
    
        IsBodiesDataFound = True
        Call CheckForUpdateInMaxMinDimensions(swView)
        
        Dim vBodies As Variant
        Dim vBodyInfo As Variant
    
        vBodies = cComp.GetBodies3(swBodyType_e.swSolidBody, vBodyInfo)
    
        If Not IsEmpty(vBodies) Then
                
            Dim i As Integer
            For i = LBound(vBodies) To UBound(vBodies)
    
                Dim swBody As SldWorks.Body2
                Set swBody = vBodies(i)

                Dim oBody As IWeldBody
                Set oBody = New IWeldBody
                                    
                oBody.Initialize cComp, swBody, swView, swNormalVector
         
                cBodiesList.AddtoList oBody
       
            Next i
    
        End If

    End If
    
     Set GetBodiesList = cBodiesList

End Function


Private Sub GetMaxMinPoint(LowerBoundPoint As Variant, HigherBoundPoint As Variant, _
            ByRef OutLower As Double, ByRef OutHigher As Double)

    If LowerBoundPoint < HigherBoundPoint Then
    
        OutLower = LowerBoundPoint
        OutHigher = HigherBoundPoint
        
    Else

        OutLower = HigherBoundPoint
        OutHigher = LowerBoundPoint
        
    End If
    
End Sub

Private Sub UpdateSheetMaxMinDimensions(swView As SldWorks.View)

    Dim MinPoint As Variant
    MinPoint = GetComponentPointInSheetSpace(cComp, cModelMinPoint, swView)

    Dim MaxPoint As Variant
    MaxPoint = GetComponentPointInSheetSpace(cComp, cModelMaxPoint, swView)
    
    Call GetMaxMinPoint(MinPoint(0), MaxPoint(0), cxMin, cxMax)
    Call GetMaxMinPoint(MinPoint(1), MaxPoint(1), cyMin, cyMax)
    Call GetMaxMinPoint(MinPoint(2), MaxPoint(2), cZMin, cZMax)
    
End Sub

Private Sub GetOriginDimensions(swView As SldWorks.View)
    
    Dim dOrginPoint(2) As Double
    dOrginPoint(0) = 0
    dOrginPoint(1) = 0
    dOrginPoint(2) = 0
    
    Dim vPoint As Variant
    vPoint = GetComponentPointInSheetSpace(cComp, dOrginPoint, swView)
    
    cxOrigin = vPoint(0)
    cyOrigin = vPoint(1)
    czOrigin = vPoint(2)
    
End Sub

Sub CheckForUpdateInMaxMinDimensions(swView As SldWorks.View)

    If IsViewChanged(swView) Then
        
        Call UpdateSheetMaxMinDimensions(swView)
        Call GetOriginDimensions(swView)
        Set cView = swView
        orgViewPosition = cView.Position
        orgScaleDecimal = cView.ScaleDecimal
        
    End If

End Sub

Private Function IsViewChanged(swView As SldWorks.View) As Boolean
    
    IsViewChanged = False

    If (swView.Name = cView.Name) Then

        Dim actualViewPosition As Variant
        actualViewPosition = swView.Position
    
        If Not (swView.ScaleDecimal = orgScaleDecimal) Or Not (orgViewPosition(0) = actualViewPosition(0)) _
                Or Not (orgViewPosition(1) = actualViewPosition(1)) Or Not (swView.Angle = cView.Angle) Then

            IsViewChanged = True
     
        End If
            
    Else
            
        IsViewChanged = True
            
    End If
    
End Function


Public Function GetCutlistFeatures(AllorNotExcludedCutlist As Boolean, Optional CutlistType As Integer = -1) As ICutlist()

    Dim swFeat As IFeature
    Set swFeat = Part.FirstFeature

    swFeat.Select2 False, -1

    Dim IsInit As Boolean
    IsInit = True

    Dim vFeats() As ICutlist

    Do While Not swFeat Is Nothing

        If swFeat.GetTypeName2 = "CutListFolder" Then

            IsExcluded = swFeat.ExcludeFromCutList

            If AllorNotExcludedCutlist Or IsExcluded = AllorNotExcludedCutlist Then

                Dim CutlistFolder As SldWorks.BodyFolder
                Set CutlistFolder = swFeat.GetSpecificFeature2

                If CutlistType = CutlistFolder.GetCutListType Then

                    Call AddFeatstoArray(IsInit, vFeats, swFeat)

                ElseIf CutlistType = -1 Then

                     Call AddFeatstoArray(IsInit, vFeats, swFeat)

                End If

            End If

        End If

        Set swFeat = swFeat.GetNextFeature

    Loop

    GetCutlistFeatures = vFeats

End Function

Private Sub AddFeatstoArray(ByRef IsInit, ByRef vFeats() As ICutlist, swSubFeat As SldWorks.Feature)

    Dim oCutlist As ICutlist
    Set oCutlist = New ICutlist
    oCutlist.Init swSubFeat, Part

    If IsInit Then

        IsInit = False
        ReDim vFeats(0)
        Set vFeats(UBound(vFeats)) = oCutlist

    Else

        ReDim Preserve vFeats(UBound(vFeats) + 1)
        Set vFeats(UBound(vFeats)) = oCutlist

    End If

End Sub

Private Function GetCutlistForThisBody(swBody As SldWorks.Body2, vCutlists As Variant) As ICutlist

    Dim i As Integer
    For i = LBound(vCutlists) To UBound(vCutlists)
    
        Dim oCutlist As ICutlist
        Set oCutlist = vCutlists(i)
        
        If oCutlist.IsContainThisBody(swBody) Then
            
            Set GetCutlistForThisBody = oCutlist
            Exit For
            
        End If
        
    Next i
    
End Function


'
'Private Function GetBodyFolder() As SldWorks.BodyFolder
'
'    Dim swFeat As IFeature
'    Set swFeat = Part.FirstFeature
'
'   While Not swFeat Is Nothing
'
'        If swFeat.GetTypeName2 = "SolidBodyFolder" Then
'
'            Set GetBodyFolder = swFeat.GetSpecificFeature2
'            Exit Function
'
'        End If
'
'        Set swFeat = swFeat.GetNextFeature
'
'    Wend
'
'End Function
'
'Function GetBOMPartNumber(ConfigName As String) As String
'
'    Dim Config As IConfiguration
'    Set Config = Model.GetConfigurationByName(ConfigName)
'
'    Select Case Config.BOMPartNoSource
'
'    Case SwConst.swBOMPartNumberSource_e.swBOMPartNumber_ConfigurationName
'
'        GetBOMPartNumber = Config.Name
'
'    Case SwConst.swBOMPartNumberSource_e.swBOMPartNumber_DocumentName
'
'        GetBOMPartNumber = GetFileName
'
'    Case SwConst.swBOMPartNumberSource_e.swBOMPartNumber_UserSpecified
'
'        GetBOMPartNumber = Config.AlternateName
'
'    Case SwConst.swBOMPartNumberSource_e.swBOMPartNumber_ParentName
'
'        Dim parentConfig As SldWorks.Configuration
'        Set parentConfig = Config.GetParent
'
'        GetBOMPartNumber = GetBOMPartNumber(parentConfig.Name)
'
'    End Select
'
'End Function
'
'Function GetFileName() As String
'
'    Dim Path As String
'    Path = Model.GetPathName()
'
'    GetFileName = Mid(Path, InStrRev(Path, "\") + 1, InStrRev(Path, ".") - InStrRev(Path, "\") - 1)
'
'End Function








