VERSION 5.00
Begin {C62A69F0-16DC-11CE-9E98-00AA00574A4F} HideShowForm 
   Caption         =   "Hide/ Show Components"
   ClientHeight    =   1440
   ClientLeft      =   108
   ClientTop       =   456
   ClientWidth     =   3408
   OleObjectBlob   =   "HideShowForm.frx":0000
   StartUpPosition =   1  'CenterOwner
End
Attribute VB_Name = "HideShowForm"
Attribute VB_GlobalNameSpace = False
Attribute VB_Creatable = False
Attribute VB_PredeclaredId = True
Attribute VB_Exposed = False

Option Explicit

Dim swSketchMgr As SldWorks.SketchManager
Dim xDirectionVector(2) As Double
Dim yDirectionVector(2) As Double
Dim zDirectionVector(2) As Double
Const SheetBorderTop As Double = 0.27030866

Private Sub CloseButton_Click()
    
    Unload Me
    
End Sub

Private Function GetOppositeVector(Dir As Variant) As Double()

    Dim Temp(2) As Double
    Dim i As Integer
    For i = LBound(Dir) To UBound(Dir)
    
        Temp(i) = -1 * Dir(i)
    
    Next i
    
    GetOppositeVector = Temp
    
End Function

Function SelectEntity(swEnt As Object, Append As Boolean, swView As SldWorks.View) As Boolean 'swView As SldWorks.View

    Dim swEntity As SldWorks.Entity
    Set swEntity = swEnt

    SelectEntity = swView.SelectEntity(swEnt, Append)
    
End Function

Private Sub CreateButton_Click()

    Me.Hide
    
    Dim wallName As String
    wallName = WallDrawingForm.WallNameComboBox.Value
    
    Dim ProjectNo As String
    ProjectNo = WallDrawingForm.ProjectNoBox.Value
    
    Unload WallDrawingForm
    
    Dim viewName As String
    viewName = GetViewName(wallName)
    
    If viewName = "" Then
    
        MsgBox "View Name not selected", vbExclamation, "Not Selected!"
        Unload Me
        Exit Sub
        
    End If

    Dim swDrawing As SldWorks.DrawingDoc
    Set swDrawing = swApp.NewDocument("C:\FBD\COMMON\FBD Templates\DEFAULT\ASSEMBLY DRAWING.drwdot", 0, 0, 0)

    Set swSketchMgr = swDrawing.SketchManager

    Dim swSheet As SldWorks.Sheet
    Set swSheet = swDrawing.GetCurrentSheet

    Call InsertSketchBlock(swDrawing, swSheet, ProjectNo)

    Dim swFrontView As SldWorks.View
    Set swFrontView = swDrawing.CreateDrawViewFromModelView3(swTopLevelModel.GetPathName(), viewName, 0.21593179, 0.15772741, 0)
    
    Dim swBottomView As SldWorks.View
    Set swBottomView = swDrawing.CreateUnfoldedViewAt3(0.21593179, 0.065, 0, False)
    
    Dim CompList As IArrListObject
    Set CompList = GetComponentsSortedWithXPosition(swFrontView, swDrawing, "EXT-")

    Dim ViewWidth As Double
    Dim ViewHeight As Double
    Dim MaxHeightComp As IComp
                
    If IsEmpty(CompList.Items) Then
    
        Set CompList = GetComponentsSortedWithXPosition(swBottomView, swDrawing, "EXT-")
        Call UpdateMaxMinPoints(CompList.Items, swFrontView)
        Call GetComponentBoundsInView(CompList, ViewWidth, ViewHeight, MaxHeightComp)
         
    Else
    
        Call GetComponentBoundsInView(CompList, ViewWidth, ViewHeight, MaxHeightComp)
        
    End If
    
    Dim IsMultipleAssembly As Boolean
    IsMultipleAssembly = CheckForMultipleAssembly(ViewWidth / swFrontView.ScaleDecimal, ViewHeight / swFrontView.ScaleDecimal)
    
    Call ActivateDrawingDocument(swTopLevelModel)
    
    Dim subAssyEndComponents As Variant
    If IsMultipleAssembly Then

        SubAssyForm.Show vbModeless

        IsSubAssyFormClicked = False
        Do While (IsSubAssyFormClicked = False)

            DoEvents

        Loop

        subAssyEndComponents = GetSelectedComponents

    End If
    
    
    InsulationForm.Show vbModeless
    IsInsulationFormClicked = False
    Do While IsInsulationFormClicked = False
        DoEvents
    Loop

    Dim vInsulationComp As Variant
    vInsulationComp = GetSelectedComponents
    
    Call ActivateDrawingDocument(swDrawing)

    Call ScaleView(swDrawing, swFrontView, ViewWidth, ViewHeight)
    
    Call UpdateMaxMinPoints(CompList.Items, swFrontView)

    Call UpdateBottomViewPosition(CompList.Items, swDrawing, swBottomView)

    Dim swLeftEdge As SldWorks.Edge
    Dim swRightEdge As SldWorks.Edge

    Dim swBottomEdge As SldWorks.Edge
    Set swBottomEdge = AddDimensionInFrontView(swFrontView, CompList.Items, MaxHeightComp, swDrawing, swLeftEdge, swRightEdge)
    

    Dim FlatCompDict As Scripting.Dictionary
    Dim CompNoDict As New Scripting.Dictionary
    Set FlatCompDict = GetCompDictionary(CompList.Items, CompNoDict)
    
    swApp.SetUserPreferenceToggle swUserPreferenceToggle_e.swSketchInference, False

    If Not IsEmpty(subAssyEndComponents) Then

        Dim vSubAssyComponentsIdx As Variant
        vSubAssyComponentsIdx = GetSubAssyComponentsIndexSorted(subAssyEndComponents, CompNoDict)

        Call AddSplitLines(vSubAssyComponentsIdx, swDrawing, swFrontView, FlatCompDict, CompNoDict, True, swLeftEdge, swRightEdge, False)
        Call AddSplitLines(vSubAssyComponentsIdx, swDrawing, swBottomView, FlatCompDict, CompNoDict, False, swLeftEdge, swRightEdge)

    End If


    Dim oSubAssy As ISubAssy
    Set oSubAssy = New ISubAssy

    Set oSubAssy.StartComp = FlatCompDict.Items(0)
    Set oSubAssy.EndComp = FlatCompDict.Items(UBound(FlatCompDict.Items))
    Set oSubAssy.StartEdge = swLeftEdge
    Set oSubAssy.EndEdge = swRightEdge
    oSubAssy.StartIdx = 0
    oSubAssy.EndIdx = UBound(FlatCompDict.Items)

    

    'Dim NoteCount As Integer
    'Call AddStructuralNotes(swDrawing, swSheet, wallName)
    
    Dim swLeftSketch As SldWorks.SketchSegment
    Dim swRightSketch As SldWorks.SketchSegment

    Call SketchLineForNonCornerPanels(swFrontView, wallName, swDrawing, oSubAssy, swBottomEdge, 0.01, swLeftSketch, swRightSketch)
    
    Dim LowerBodies As IArrListObject
    Set LowerBodies = New IArrListObject
    
    Dim UpperBodies As IArrListObject
    Set UpperBodies = New IArrListObject
    
    Dim MiddleBodies As IArrListObject
    Set MiddleBodies = New IArrListObject
    

    Dim Clearance As Double
    
    If Not IsEmpty(vInsulationComp) Then
        
        Call GetSolidBodyList(vInsulationComp, swFrontView, swDrawing, LowerBodies, UpperBodies, MiddleBodies) ' oSubAssy.yMax, oSubAssy.yMin, Clearance)


        Call AddInsulationMaterialNote(UpperBodies, swBottomView, swDrawing)
        Call AddInsulationHatches(vInsulationComp, swBottomView, swDrawing)

        Call AddCrossMarkForAssyCuts(vInsulationComp, swFrontView, swDrawing)
        Call AddCrossMarkForComponentCuts(vInsulationComp, swFrontView, swDrawing)
        
        Call AddCallouts(UpperBodies.Items, swDrawing, swFrontView, oSubAssy.yMin, oSubAssy.yMax, True, False, Clearance)
        Call AddCallouts(MiddleBodies.Items, swDrawing, swFrontView, oSubAssy.yMin, oSubAssy.yMax, False, False, Clearance)
        
        Call AddCallouts(LowerBodies.Items, swDrawing, swFrontView, oSubAssy.yMin, oSubAssy.yMax, False, True, Clearance)
        
        If Not IsEmpty(LowerBodies.Items) Then
        
            Clearance = Clearance + 0.00625
            
        Else
            
            Clearance = 0.005
            
        End If
        
        Call AddDimensionFromEnd(UpperBodies.Items, swLeftSketch, oSubAssy.StartEdge, swFrontView, swDrawing, oSubAssy.StartComp.yMin - Clearance)
        Call AddDimensionFromEnd(UpperBodies.Items, swRightSketch, oSubAssy.EndEdge, swFrontView, swDrawing, oSubAssy.StartComp.yMin - Clearance, False)

'
    End If
    
    Clearance = Clearance + 0.005
    Call AddOverallDimension(oSubAssy, swDrawing, swFrontView, Clearance)
    Call AddDimensionFromCornerSketches(swDrawing, swFrontView, swLeftSketch, swRightSketch, oSubAssy, Clearance)

    Call CleanUpActivateAndAddViewLabel(swDrawing, swFrontView, wallName, oSubAssy.StartComp.yMin - Clearance - 0.005, (oSubAssy.StartComp.xMin + oSubAssy.EndComp.xMax) / 2)

    
    Call AddCastingSketchAndNote(oSubAssy.EndComp, swBottomView, swSketchMgr, swDrawing)
    Call UpdateFrontViewPosition(CompList.Items, swDrawing, swFrontView)

    swApp.SetUserPreferenceToggle swUserPreferenceToggle_e.swSketchInference, True

    Unload Me
    
End Sub

Private Sub GetSolidBodyList(vComps As Variant, swView As SldWorks.View, swDrawing As SldWorks.DrawingDoc, _
                ByRef LowerBodies As IArrListObject, ByRef UpperBodies As IArrListObject, ByRef MiddleBodies As IArrListObject)

    
    Dim i As Integer
    For i = LBound(vComps) To UBound(vComps)
    
        Dim swComp As SldWorks.Component2
        Set swComp = vComps(i)
        
        swComp.ExcludeFromBOM = False
        
        Dim vCompBounds As Variant
        vCompBounds = swComp.GetModelDoc2().GetPartBox(True)
    
        Dim vLowerCompBound(2) As Double
        vLowerCompBound(0) = vCompBounds(0)
        vLowerCompBound(1) = vCompBounds(1)
        vLowerCompBound(2) = vCompBounds(2)
    
        Dim LowerCompBound As Variant
        LowerCompBound = GetComponentPointInSheetSpace(swComp, vLowerCompBound, swView)
    
        Dim vUpperCompBound(2) As Double
        vUpperCompBound(0) = vCompBounds(3)
        vUpperCompBound(1) = vCompBounds(4)
        vUpperCompBound(2) = vCompBounds(5)
    
        Dim UpperCompBound As Variant
        UpperCompBound = GetComponentPointInSheetSpace(swComp, vUpperCompBound, swView)
    
        Dim yCompMin As Double
        Dim yCompMax As Double
        Call GetMaxMinPoint(LowerCompBound(1), UpperCompBound(1), yCompMin, yCompMax)

        Dim vBodies As Variant
        Dim vBodiesInfo As Variant
    
        vBodies = swComp.GetBodies3(swBodyType_e.swSolidBody, vBodiesInfo)
        
        If Not IsEmpty(vBodies) Then

            Dim j As Integer
            For j = LBound(vBodies) To UBound(vBodies)
    
                Dim swBody As SldWorks.Body2
                Set swBody = vBodies(j)
    
                Dim oBody As ISolidBody
                Set oBody = New ISolidBody
    
                oBody.Initialize swBody, swComp, swView
    
                If Abs(oBody.yMax - yCompMax) <= 0.001 Then
                
                    UpperBodies.AddtoList oBody
    
    
                ElseIf Abs(oBody.yMin - yCompMin) <= 0.001 Then
                    
                    LowerBodies.AddtoList oBody
    
                Else
                    
                    MiddleBodies.AddtoList oBody
    
                End If
    
            Next j
            
        End If
        
    Next i
    
    UpperBodies.SortItems "xMin", False
    LowerBodies.SortItems "xMin", False
    MiddleBodies.SortItems "xMin", False

End Sub

Private Sub UpdateFrontViewPosition(vComps As Variant, swDrawing As SldWorks.DrawingDoc, swView As SldWorks.View)
    
    Dim oStartComp As IComp
    Set oStartComp = vComps(0)
    
    Dim oEndComp As IComp
    Set oEndComp = vComps(UBound(vComps))
    
    Dim CenterX As Double
    CenterX = (oStartComp.xMin + oEndComp.xMax) / 2

    Dim viewPosition As Variant
    viewPosition = swView.Position

    viewPosition(0) = viewPosition(0) + (viewPosition(0) - CenterX)

    swView.Position = viewPosition
    
End Sub

Sub AddCrossMarkForAssyCuts(vComps As Variant, swView As SldWorks.View, swDrawing As SldWorks.DrawingDoc)
                
    If Not IsEmpty(vComps) Then
    
        Dim FullAssyName As String
        FullAssyName = Replace(vComps(0).Name2, "/" & _
                    Right(vComps(0).Name2, Len(vComps(0).Name2) - InStrRev(vComps(0).Name2, "/")), "")
        
        Dim AssyName As String
        AssyName = Right(FullAssyName, Len(FullAssyName) - InStrRev(FullAssyName, "/"))
                    
        Dim swTopLevelAssy As SldWorks.AssemblyDoc
        Set swTopLevelAssy = swTopLevelModel

        Dim swWallAssy As SldWorks.AssemblyDoc
        Set swWallAssy = swTopLevelAssy.GetComponentByName(AssyName).GetModelDoc2()
        
        Dim Errors As Long
        swApp.ActivateDoc3 swWallAssy.GetPathName, True, swRebuildOnActivation_e.swDontRebuildActiveDoc, Errors
        
        Dim vAssyCutFeatures As Variant
        vAssyCutFeatures = GetAssyCutFeaturesIfAny(vComps, swWallAssy)
          
        Call GetContoursAndAddCrossMark(vAssyCutFeatures, swDrawing, swView, FullAssyName)

        swApp.CloseDoc swWallAssy.GetPathName
        
    End If
    

End Sub

Sub AddCrossMarkForComponentCuts(vComps As Variant, swView As SldWorks.View, swDrawing As SldWorks.DrawingDoc)
                
    If Not IsEmpty(vComps) Then
    
        Dim i As Integer
        For i = LBound(vComps) To UBound(vComps)
        
            Dim swInsulationComp As SldWorks.Component2
            Set swInsulationComp = vComps(i)
            
            Debug.Print swInsulationComp.Name2
        
            Dim vCompCutFeatures As Variant
            vCompCutFeatures = GetComponentCutFeaturesIfAny(swInsulationComp)
        
            Call GetContoursAndAddCrossMark(vCompCutFeatures, swDrawing, swView, swInsulationComp.Name2)
            
        Next i
        
    End If
    

End Sub

Function GetComponentCutFeaturesIfAny(swComp As SldWorks.Component2) As Variant

    Dim CutFeaturesDict As Scripting.Dictionary
    Set CutFeaturesDict = New Scripting.Dictionary

    Dim swFeatManager As SldWorks.FeatureManager
    Set swFeatManager = swComp.GetModelDoc2().FeatureManager
    
    Dim vFeats As Variant
    vFeats = swFeatManager.GetFeatures(True)
    
    Dim i As Integer
    For i = UBound(vFeats) To LBound(vFeats) Step -1
    
        Dim swFeat As SldWorks.Feature
        Set swFeat = vFeats(i)
        
        Debug.Print swFeat.Name
        Debug.Print swFeat.GetTypeName2
        
        Dim vSuppressed As Variant
        vSuppressed = swFeat.IsSuppressed2(swInConfigurationOpts_e.swThisConfiguration, Nothing)
        
        If False = vSuppressed(0) Then
        
            If (swFeat.GetTypeName2 = "ICE" And InStr(swFeat.Name, "Cut") > 0) Or swFeat.GetTypeName2 = "Cut" Then

                If Not CutFeaturesDict.Exists(swFeat.Name) Then
                    
                    CutFeaturesDict.Add swFeat.Name, swFeat
                        
                End If
                
            End If
            
        End If

    Next i
    
    
    GetComponentCutFeaturesIfAny = CutFeaturesDict.Items

End Function


Sub GetContoursAndAddCrossMark(vCutFeatures As Variant, swDrawing As SldWorks.DrawingDoc, _
        swView As SldWorks.View, AssyName As String)

    If Not IsEmpty(vCutFeatures) Then
    
        Dim i As Integer
        For i = LBound(vCutFeatures) To UBound(vCutFeatures)
        
            Dim swFeat As SldWorks.Feature
            Set swFeat = vCutFeatures(i)
            
            Dim swSubFeat As SldWorks.Feature
            Set swSubFeat = swFeat.GetFirstSubFeature
            
            If swSubFeat.GetTypeName2 = "ProfileFeature" Then
            
                Dim swSketch As SldWorks.Sketch
                Set swSketch = swSubFeat.GetSpecificFeature2
                
                Dim vContourArrList As New IArrListObject
                Set vContourArrList = GetSketchContours(swSketch)
                
                Dim vContours As Variant
                vContours = vContourArrList.Items
            
                If Not (IsEmpty(vContours)) Then
                
                    Call AddCrossMarkForContours(vContours, swDrawing, swSubFeat, swSketch, swView, AssyName)
                    
                End If
            
            End If

        Next i
        
    End If
End Sub

Function GetAssyCutFeaturesIfAny(vComps As Variant, swWallAssy As SldWorks.AssemblyDoc) As Variant

    Dim CutFeaturesDict As Scripting.Dictionary
    Set CutFeaturesDict = New Scripting.Dictionary

    Dim swFeatManager As SldWorks.FeatureManager
    Set swFeatManager = swWallAssy.FeatureManager
    
    Dim vFeats As Variant
    vFeats = swFeatManager.GetFeatures(True)
    
    Dim i As Integer
    For i = UBound(vFeats) To LBound(vFeats) Step -1
    
        Dim swFeat As SldWorks.Feature
        Set swFeat = vFeats(i)
        
        Debug.Print swFeat.Name
        Debug.Print swFeat.GetTypeName2
        
        If False = swFeat.IsSuppressed Then
        
            If swFeat.GetTypeName2 = "Cut" Then
    
                If IsFeatureAffectAnyComp(vComps, swWallAssy, swFeat) Then
            
                    If Not CutFeaturesDict.Exists(swFeat.Name) Then
                    
                        CutFeaturesDict.Add swFeat.Name, swFeat
                        
                    End If
                    
                End If
                
            End If
            
        End If
        
        If swFeat.GetTypeName2 = "MateGroup" Then
        
            Exit For
            
        End If
        
    Next i
    
    GetAssyCutFeaturesIfAny = CutFeaturesDict.Items

End Function

Function IsFeatureAffectAnyComp(vInsulationComps As Variant, _
            swWallAssy As SldWorks.AssemblyDoc, swFeat As SldWorks.Feature) As Boolean
            
    IsFeatureAffectAnyComp = False
      
    Dim vComps As Variant
    vComps = swWallAssy.GetFeatureScope(swFeat)
    
    Dim i As Integer
    For i = LBound(vComps) To UBound(vComps)
    
        Dim swFeatAffectedComp As SldWorks.Component2
        Set swFeatAffectedComp = vComps(i)
        
        'Debug.Print swFeatAffectedComp.Name2
        
        Dim j As Integer
        For j = LBound(vInsulationComps) To UBound(vInsulationComps)
        
            Dim swInsulationComp As SldWorks.Component2
            Set swInsulationComp = vInsulationComps(j)
            'Debug.Print swInsulationComp.Name2
        
            If InStr(swInsulationComp.Name2, swFeatAffectedComp.Name2) > 0 Then
                
                IsFeatureAffectAnyComp = True
                Exit Function
                
            End If
            
        Next j

    Next i

End Function


Sub AddInsulationMaterialNote(SolidBodyList As IArrListObject, _
        swView As SldWorks.View, swDrawing As SldWorks.ModelDoc2)
    
    
    swDrawing.ActivateView swView.Name
    
    Dim oBodyInFrontView As ISolidBody
    Set oBodyInFrontView = SolidBodyList.Items(UBound(SolidBodyList.Items))
    
    Dim MaterialName As String
    MaterialName = oBodyInFrontView.GetComponent.GetModelDoc2().MaterialIdName
    
    If MaterialName = "" Then
    
        MaterialName = "INSULATION"
        
    End If
    
    Dim StringPos As Integer
    StringPos = InStr(MaterialName, "|")
    
    If StringPos > 0 Then

        MaterialName = Right(MaterialName, Len(MaterialName) - StringPos)

    End If
    


    Dim oEndBody As ISolidBody
    Set oEndBody = New ISolidBody
    
    oEndBody.Initialize oBodyInFrontView.GetBody, oBodyInFrontView.GetComponent, swView
    
    Dim vFaces As Variant
    vFaces = swView.GetVisibleEntities2(oBodyInFrontView.GetComponent, swViewEntityType_e.swViewEntityType_Face)
    
    Dim IsSelected As Boolean
    Dim xPos As Double
    xPos = (oEndBody.xMax + oEndBody.xMin) / 2
    
    Dim yPos As Double
    yPos = (oEndBody.yMax + oEndBody.yMin) / 2
    
    swDrawing.ViewZoomTo2 oEndBody.xMin, oEndBody.yMin, oEndBody.zMin, oEndBody.xMax, oEndBody.yMax, oEndBody.zMax
    swDrawing.ClearSelection2 True
    
    IsSelected = SelectFaceWithPosition(swDrawing, oEndBody, xPos, yPos, CheckComp:=True)
    
    swDrawing.ViewZoomTo2 0, 0, 0, 17 * 0.0254, 11 * 0.0254, 0
    
    If False = IsSelected Then
        
        IsSelected = SelectFaceOfTheBody(vFaces, oEndBody, swDrawing, swView, False)
        xPos = oEndBody.xMax
        
        If False = IsSelected Then
            
            swView.SelectEntity vFaces(0), False
            
        End If
        
    End If
    
    If IsSelected Then
    
        Dim swAnn As SldWorks.Annotation
        Set swAnn = AddNoteToView(swDrawing, UCase(MaterialName), xPos - Len(MaterialName) * 0.002, yPos + 0.008)
        
        swAnn.SetLeader3 swLeaderStyle_e.swBENT, swLeaderSide_e.swLS_SMART, False, False, True, False
        
        Dim HeadStyle As Integer
        HeadStyle = swAnn.SetArrowHeadStyleAtIndex(0, swArrowStyle_e.swCLOSED_ARROWHEAD)
        
    End If


End Sub

Sub UpdateHatchProperties(swView As SldWorks.View)
    
    Dim swSketch As SldWorks.Sketch
    Set swSketch = swView.GetSketch
    
    Dim vSketchHatches As Variant
    vSketchHatches = swSketch.GetSketchHatches
            
    If Not IsEmpty(vSketchHatches) Then
            
        Dim i As Integer
        For i = LBound(vSketchHatches) To UBound(vSketchHatches)
                
            Dim swSketchHatch As SldWorks.SketchHatch
            Set swSketchHatch = vSketchHatches(i)
                    
            swSketchHatch.Pattern = "ISO (Steel)"
            swSketchHatch.Scale2 = swView.ScaleDecimal * 4
                
        Next i
                
    End If

End Sub

Private Sub AddInsulationHatches(vComps As Variant, swView As SldWorks.View, swDrawing As SldWorks.DrawingDoc)
    
    swDrawing.ClearSelection2 True
    
    Dim j As Integer
    For j = LBound(vComps) To UBound(vComps)
        
        Dim swComp As SldWorks.Component2
        Set swComp = vComps(j)
    
        Dim vFaces As Variant
        vFaces = swView.GetVisibleEntities2(swComp, swViewEntityType_e.swViewEntityType_Face)
        
        If Not IsEmpty(vFaces) Then
        
            Dim i As Integer
            For i = LBound(vFaces) To UBound(vFaces)
        
                Dim swFace As SldWorks.Face2
                Set swFace = vFaces(i)
                
                swView.SelectEntity swFace, True
                
            Next i
            
            swDrawing.InsertHatchedFace
            Call UpdateHatchProperties(swView)
            
        End If
        
    Next j

End Sub




'Private Function GetSolidBodyList(swComp As SldWorks.Component2, swView As SldWorks.View, swDrawing As SldWorks.DrawingDoc) As IArrListObject 'OLD
'
'    swDrawing.ActivateView swView.Name
'
'    Set GetSolidBodyList = New IArrListObject
'

'
'    Dim vBodies As Variant
'    Dim vBodiesInfo As Variant
'
'    vBodies = swComp.GetBodies3(swBodyType_e.swSolidBody, vBodiesInfo)
'
'    Dim vFaces As Variant
'    vFaces = swView.GetVisibleEntities2(swComp, swViewEntityType_e.swViewEntityType_Face)
'
'    If Not IsEmpty(vBodies) Then
'
'        Dim i As Integer
'        For i = LBound(vBodies) To UBound(vBodies)
'
'            Dim swBody As SldWorks.Body2
'            Set swBody = vBodies(i)
'
'            Dim oBody As ISolidBody
'            Set oBody = New ISolidBody
'
'            oBody.Initialize swBody, swComp, swView
'
'            GetSolidBodyList.AddtoList oBody
'
'            Dim xPos As Double
'            Dim yPos As Double
'
'            Dim annYPos As Double
'
'            xPos = (oBody.xMin + oBody.xMax) / 2
'            yPos = (oBody.yMin + oBody.yMax) / 2
'            annYPos = yPos '0.9 * oBody.yMax - 0.1 * oBody.yMin
'
'            Dim IsSelected As Boolean
'
'
'            If Abs(oBody.xMax - oBody.xMin) <= 0.01 Or Abs(oBody.yMax - oBody.yMin) <= 0.01 Then
'
'                yPos = 0.7 * oBody.yMax + 0.3 * oBody.yMin
'                annYPos = yPos + 0.01
'                IsSelected = SelectFaceWithPosition(swDrawing, oBody, xPos, yPos)
'
'            Else
'
'                IsSelected = SelectFaceOfTheBody(vFaces, oBody, swDrawing, swView, False)
'
'            End If
'

'
'                End If
'
'            End If
'
'        Next i
'
'        GetSolidBodyList.SortItems "xMin", False
'
'    End If
'
'End Function

'Private Function GetSolidBodyList(swComp As SldWorks.Component2, swView As SldWorks.View, swDrawing As SldWorks.DrawingDoc, _
'                yAssyMax As Double, yAssyMin As Double, ByRef Clearance As Double) As IArrListObject
'
'    swDrawing.ActivateView swView.Name
'
'    Set GetSolidBodyList = New IArrListObject
'
'    Dim swBalloonParams As SldWorks.BalloonOptions
'    Set swBalloonParams = swDrawing.Extension.CreateBalloonOptions()
'    swBalloonParams.Size = swBalloonFit_e.swBF_Tightest
'    swBalloonParams.Style = swBalloonStyle_e.swBS_Inspection
'    swBalloonParams.UpperTextContent = swBalloonTextContent_e.swBalloonTextCutlistProperties
'    swBalloonParams.UpperText = "$PRPWLD:" & Chr(34) & "PART CODE" & Chr(34) '"$PRPWLD:" & Chr(34) & "ITEM NO" & Chr(34)
'
'    Dim vBodies As Variant
'    Dim vBodiesInfo As Variant
'
'    vBodies = swComp.GetBodies3(swBodyType_e.swSolidBody, vBodiesInfo)
'
'    Dim vFaces As Variant
'    vFaces = swView.GetVisibleEntities2(swComp, swViewEntityType_e.swViewEntityType_Face)
'
'    Dim vCompBounds As Variant
'    vCompBounds = swComp.GetBox(False, False)
'
'    Dim vLowerCompBound(2) As Double
'    vLowerCompBound(0) = vCompBounds(0)
'    vLowerCompBound(1) = vCompBounds(1)
'    vLowerCompBound(2) = vCompBounds(2)
'
'    Dim LowerCompBound As Variant
'    LowerCompBound = GetComponentPointInSheetSpace(swComp, vLowerCompBound, swView)
'
'    Dim vUpperCompBound(2) As Double
'    vUpperCompBound(0) = vCompBounds(3)
'    vUpperCompBound(1) = vCompBounds(4)
'    vUpperCompBound(2) = vCompBounds(5)
'
'    Dim UpperCompBound As Variant
'    UpperCompBound = GetComponentPointInSheetSpace(swComp, vUpperCompBound, swView)
'
'    Dim yCompMin As Double
'    Dim yCompMax As Double
'    Call GetMaxMinPoint(LowerCompBound(1), UpperCompBound(1), yCompMin, yCompMax)
'
'    If Not IsEmpty(vBodies) Then
'
'        Dim i As Integer
'        For i = LBound(vBodies) To UBound(vBodies)
'
'            Dim swBody As SldWorks.Body2
'            Set swBody = vBodies(i)
'
'            Dim oBody As ISolidBody
'            Set oBody = New ISolidBody
'
'            oBody.Initialize swBody, swComp, swView
'
'            GetSolidBodyList.AddtoList oBody
'
'            Dim xPos As Double
'            Dim yPos As Double
'
'            Dim annYPos As Double
'
'            xPos = (oBody.xMin + oBody.xMax) / 2
'
'
'            If Abs(oBody.yMax - yCompMax) <= 0.001 Then
'
'                yPos = 0.9 * oBody.yMax + 0.1 * oBody.yMin
'                annYPos = yAssyMax + 0.005
'
'            ElseIf Abs(oBody.yMin - yCompMin) <= 0.001 Then
'
'                If Clearance = 0 Then
'
'                    Clearance = 0.005
'
'                End If
'
'                yPos = 0.1 * oBody.yMax + 0.9 * oBody.yMin
'                annYPos = yAssyMin - Clearance
'
'            Else
'
'                yPos = (oBody.yMin + oBody.yMax) / 2
'                annYPos = oBody.yMin - 0.01
'
'            End If
'
'            Dim IsSelected As Boolean
'            IsSelected = SelectFaceWithPosition(swDrawing, oBody, xPos, yPos)
'
'            If False = IsSelected Then
'
'                IsSelected = SelectFaceOfTheBody(vFaces, oBody, swDrawing, swView, False)
'
'            End If
'
''            If Abs(oBody.xMax - oBody.xMin) <= 0.01 Or Abs(oBody.yMax - oBody.yMin) <= 0.01 Then
''
''                yPos = 0.7 * oBody.yMax + 0.3 * oBody.yMin
''                annYPos = yPos + 0.01
''
''
''            Else
''
''                IsSelected = SelectFaceOfTheBody(vFaces, oBody, swDrawing, swView, False)
''
''            End If
'
'            If IsSelected Then
'
'                Dim swNote As SldWorks.Note
'                Set swNote = swDrawing.Extension.InsertBOMBalloon2(swBalloonParams)
'
'                swNote.PropertyLinkedText = "$PRPWLD:" & Chr(34) & "PART CODE" & Chr(34)
'
'                If Not swNote Is Nothing Then
'
'                    Dim swAnn As SldWorks.Annotation
'                    Set swAnn = swNote.GetAnnotation
'
'                    swAnn.SetPosition2 xPos, annYPos, 0
'
''                    If (yPos = annYPos) Then
''
''                        swAnn.SetLeader3 swLeaderStyle_e.swNO_LEADER, swLeaderSide_e.swLS_SMART, False, False, True, False
''
''                    End If
'
'                End If
'
'            End If
'
'        Next i
'
'        GetSolidBodyList.SortItems "xMin", False
'
'    End If
'
'End Function

Private Function SelectFaceOfTheBody(vFaces As Variant, oBody As ISolidBody, swDrawing As SldWorks.DrawingDoc, _
                    swView As SldWorks.View, Append As Boolean) As Boolean
    
    If Not IsEmpty(vFaces) Then
    
        Dim i As Integer
        For i = LBound(vFaces) To UBound(vFaces)
        
            Dim swFace As SldWorks.Face2
            Set swFace = vFaces(i)
            
            Dim swFaceBody As SldWorks.Body2
            Set swFaceBody = swFace.GetBody
            
            If swFaceBody.Name = oBody.GetBody.Name Then
            
                SelectFaceOfTheBody = swView.SelectEntity(swFace, Append)
                Exit For
                
            End If
    
        Next i
        
'    Else
'
'        SelectFaceOfTheBody = SelectFaceWithPosition(swDrawing, oBody, (oBody.xMin + oBody.xMax) / 2, (oBody.yMin + oBody.yMax) / 2)

    End If

End Function

Private Function SelectFaceWithPosition(swDrawing As SldWorks.DrawingDoc, oBody As ISolidBody, xPos As Double, _
    yPos As Double, Optional Append As Boolean = False, Optional CheckComp As Boolean = False) As Boolean

    SelectFaceWithPosition = swDrawing.Extension.SelectByID2("", "FACE", xPos, yPos, _
                    0, Append, -1, Nothing, 1)
                    
    If SelectFaceWithPosition Then

        Dim swSelectMgr As SldWorks.SelectionMgr
        Set swSelectMgr = swDrawing.SelectionManager

        Dim swCompCheck As SldWorks.DrawingComponent
        Set swCompCheck = swSelectMgr.GetSelectedObjectsComponent4(2, -1)
        
        Dim swCompFace As SldWorks.Face2
        Set swCompFace = swSelectMgr.GetSelectedObject6(2, -1)
        
        If CheckComp Then
        
            If Not (Right(swCompCheck.Name, Len(swCompCheck.Name) - InStrRev(swCompCheck.Name, "/")) = _
                    Right(oBody.GetComponent.Name2, Len(oBody.GetComponent.Name2) - InStrRev(oBody.GetComponent.Name2, "/"))) Then
                
                SelectFaceWithPosition = False
                swDrawing.ClearSelection2 True
                
            End If
            
        Else
        
            If Not (swCompFace.GetBody.Name = oBody.GetBody.Name) Then
            
                SelectFaceWithPosition = False
                swDrawing.ClearSelection2 True

            End If
            
        End If
        
    End If

End Function

Private Sub UpdateMaxMinPoints(vComps As Variant, swView As SldWorks.View)

    Dim i As Integer
    For i = LBound(vComps) To UBound(vComps)
    
        Dim oComp As IComp
        Set oComp = vComps(i)

        Dim MinPoint As Variant
        MinPoint = GetComponentPointInSheetSpace(oComp.GetComponent, oComp.GetMinPointInModel, swView)
    
        Dim MaxPoint As Variant
        MaxPoint = GetComponentPointInSheetSpace(oComp.GetComponent, oComp.GetMaxPointInModel, swView)
        
        oComp.UpdateSheetMaxMinDimensions swView, MinPoint, MaxPoint
        
    Next i

End Sub
 
Private Sub AddCastingSketchAndNote(oComp As IComp, swView As SldWorks.View, swSketchMgr As SldWorks.SketchManager, _
                swDrawing As SldWorks.DrawingDoc)
    
    swDrawing.ActivateView swView.Name
    
    Dim xMin As Double
    Dim yMin As Double
    Dim xMax As Double
    Dim yMax As Double
        
    Call GetViewMaxMinPoints(oComp, swView, xMin, xMax, yMin, yMax)
   
    Dim swSketchSegment As SldWorks.SketchSegment
    Set swSketchSegment = swSketchMgr.CreateLine(xMax - 0.25 * 0.0254, yMin, _
                                0, xMax + 16 * 0.0254, yMin, 0)
                                
    swSketchSegment.ConstructionGeometry = True
        
    Dim vSketchPoint As Variant
    vSketchPoint = SelectSketchSegment(swSketchSegment, swDrawing, swView, False, True, 0.5)
    Call AddNoteToView(swDrawing, "CASTING BED", vSketchPoint(0) - 0.02, vSketchPoint(1) - 0.01)
        
    Dim swEdge As SldWorks.Edge
    Set swEdge = GetEdgeInView(oComp, swView, True, False)
        
    Call AddCollinearRelation(swDrawing, swEdge, swSketchSegment, swView)
    
End Sub

Function AddSubAssyComponentsToDictionary(vComps As Variant) As Scripting.Dictionary

    Set AddSubAssyComponentsToDictionary = New Scripting.Dictionary
    
    If Not IsEmpty(vComps) Then
    
        Dim i As Integer
        For i = LBound(vComps) To UBound(vComps)
            
            If Not AddSubAssyComponentsToDictionary.Exists(vComps(i).Name2) Then
            
                AddSubAssyComponentsToDictionary.Add vComps(i).Name2, vComps
                
            End If
        
        Next i
    
    End If
    
End Function

Sub CheckandAddLayer(LayName As String, LayerDesc As String, swLayerMgr As SldWorks.LayerMgr)

    Dim vLayNames As Variant
    vLayNames = swLayerMgr.GetLayerList
    
    Dim IsLayerExists As Boolean
    
    Dim i As Integer
    For i = 0 To UBound(vLayNames)
    
        If vLayNames(i) = LayName Then
        
            IsLayerExists = True
            Exit For
            
        End If
        
    Next i
    
    If Not (IsLayerExists) Then
    
        swLayerMgr.AddLayer LayName, LayerDesc, 0, swLineStyles_e.swLineDEFAULT, swLineWeights_e.swLW_NONE
        
        Dim swlayer As SldWorks.Layer
        Set swlayer = swLayerMgr.GetLayer(LayName)
        
        swlayer.Style = swLineStyles_e.swLineCENTER
        swlayer.Width = swLineWeights_e.swLW_THICK5
        
    End If
    
End Sub

Private Sub SketchLineForNonCornerPanels(swView As SldWorks.View, wallName As String, _
        swDrawing As SldWorks.ModelDoc2, oSubAssy As ISubAssy, swBottomEdge As SldWorks.Edge, _
            ByRef MaxClearance As Double, ByRef swStartSketch As SldWorks.SketchSegment, ByRef swEndSketch As SldWorks.SketchSegment)
    
    swDrawing.ActivateView swView.Name
    
    If InStr(wallName, "Wall") > 0 And (InStr(wallName, "-A") > 0 Or InStr(wallName, "-B") Or _
                                InStr(wallName, "-C") Or InStr(wallName, "-D")) Then
    
        Dim viewDrawComp As SldWorks.DrawingComponent
        Set viewDrawComp = swView.RootDrawingComponent
        
        Dim viewComp As SldWorks.Component2
        Set viewComp = viewDrawComp.Component
        
        Debug.Print viewComp.Name2
    
        Dim swControlSketch As SldWorks.Component2
        Set swControlSketch = GetControlSketch
        
        Dim PlaneName As String
        
        If Not InStr(oSubAssy.StartComp.GetCustomProperty("Profile"), "CORNER") > 0 Then
        
            PlaneName = GetPlaneName(wallName, True)
            Set swStartSketch = CreateSketchLinesForNonCornerPanels(PlaneName, viewDrawComp, swControlSketch, viewComp, oSubAssy.StartComp, swView, swDrawing)
            Call AddSplitLineNote(swStartSketch, swDrawing, swView, "EXTERNAL WALL-" & Right(PlaneName, 1), False, 0.035)
        
        End If
        
        If Not InStr(oSubAssy.EndComp.GetCustomProperty("Profile"), "CORNER") > 0 Then
        
            PlaneName = GetPlaneName(wallName, False)
            Set swEndSketch = CreateSketchLinesForNonCornerPanels(PlaneName, viewDrawComp, swControlSketch, viewComp, oSubAssy.EndComp, swView, swDrawing)
            Call AddSplitLineNote(swEndSketch, swDrawing, swView, "EXTERNAL WALL-" & Right(PlaneName, 1))
            
        End If
        
        
        

    End If
    

End Sub

Private Sub AddDimensionFromCornerSketches(swDrawing As SldWorks.DrawingDoc, swView As SldWorks.View, _
                swStartSketch As SldWorks.SketchSegment, swEndSketch As SldWorks.SketchSegment, _
            oSubAssy As ISubAssy, ByRef MaxClearance As Double)

    Dim swDisplayDim As SldWorks.DisplayDimension
        
        If Not swStartSketch Is Nothing Then
        
            MaxClearance = MaxClearance + 0.006
            
            If Not swEndSketch Is Nothing Then
                
                swStartSketch.Select4 False, Nothing
                swEndSketch.Select4 True, Nothing

            Else
            
                swStartSketch.Select4 False, Nothing
                Call SelectEntity(oSubAssy.EndEdge, True, swView)
    
            End If
            
        Else

            If Not swEndSketch Is Nothing Then
            
                MaxClearance = MaxClearance + 0.006
                swEndSketch.Select4 False, Nothing
                Call SelectEntity(oSubAssy.StartEdge, True, swView)
            
            End If
            
        End If
        
        Set swDisplayDim = swDrawing.AddHorizontalDimension2(oSubAssy.StartComp.xMin + 0.01, oSubAssy.EndComp.yMin - MaxClearance, 0)
        If Not swDisplayDim Is Nothing Then

            swDisplayDim.CenterText = True
            swDisplayDim.SetDual2 False, False
            
        End If
End Sub



Private Sub AddDimensionFromEnd(vSolidBodies As Variant, swSketchLine As SldWorks.SketchSegment, _
        swEdge As SldWorks.Edge, swView As SldWorks.View, swDrawing As SldWorks.DrawingDoc, _
        yMin As Double, Optional IsStart As Boolean = True)
        
    swDrawing.ActivateView swView.Name
        
    If Not IsEmpty(vSolidBodies) Then
        
        If swSketchLine Is Nothing Then
            
            Call SelectEntity(swEdge, False, swView)
                
        Else
        
            swSketchLine.Select4 False, Nothing
                
        End If
        
        Dim swBodyEdge As SldWorks.Edge
        Dim oBody As ISolidBody
        Dim xPos As Double
        
        If IsStart Then
        
            Set oBody = vSolidBodies(0)
            xPos = oBody.xMin + 0.015
            Set swBodyEdge = GetEdgeInViewForBody(oBody.GetComponent, oBody, swView, False, False)
            
        Else
        
            
            Set oBody = vSolidBodies(UBound(vSolidBodies))
            xPos = oBody.xMax - 0.015
            Set swBodyEdge = GetEdgeInViewForBody(oBody.GetComponent, oBody, swView, False, True)
            
        End If
        
        Call SelectEntity(swBodyEdge, True, swView)
        
        Dim swDisplayDim As SldWorks.DisplayDimension
        Set swDisplayDim = swDrawing.AddHorizontalDimension2(xPos, yMin, 0)
        
        If Not swDisplayDim Is Nothing Then

            swDisplayDim.CenterText = True
            swDisplayDim.ShowParenthesis = True
            
        End If
        
    End If
            
    
End Sub

Private Function CreateSketchLinesForNonCornerPanels(PlaneName As String, viewDrawComp As SldWorks.DrawingComponent, swControlSketch As SldWorks.Component2, _
                                        viewComp As SldWorks.Component2, oComp As IComp, swView As SldWorks.View, _
                                        swDrawing As SldWorks.ModelDoc2) As SldWorks.SketchSegment
                    
        Dim xMin As Double
        Dim yMin As Double
        Dim xMax As Double
        Dim yMax As Double
        Call GetViewMaxMinPoints(oComp, swView, xMin, xMax, yMin, yMax)
        
        Dim swSketchSegment As SldWorks.SketchSegment
        Set swSketchSegment = swSketchMgr.CreateLine(xMax, yMax + 16 * 0.0254, _
                                    0, xMax, yMin - 16 * 0.0254, 0)
        swSketchSegment.ConstructionGeometry = True
            
        swDrawing.Extension.SelectByID2 PlaneName & "@" & viewDrawComp.Name & "@" & swView.Name & "/" & swControlSketch.Name & "@" & viewComp.Name2, "PLANE", 0, 0, 0, False, 0, Nothing, 0
        swSketchSegment.Select4 True, Nothing
            
        swDrawing.SketchAddConstraints "sgCOLINEAR"
        
        Set CreateSketchLinesForNonCornerPanels = swSketchSegment

End Function

Private Function GetPlaneName(wallName As String, IsLeftPanel As Boolean) As String

    If IsLeftPanel Then
        
        Select Case wallName
            
            Case "Wall-A"
            
                GetPlaneName = "Outside D"
            
            Case "Wall-B"
            
                GetPlaneName = "Outside A"
            
            Case "Wall-C"
            
                GetPlaneName = "Outside B"
            
            Case "Wall-D"
            
                GetPlaneName = "Outside C"

        End Select
        
    Else
        
        Select Case wallName
            
            Case "Wall-A"
            
                GetPlaneName = "Outside B"
            
            Case "Wall-B"
            
                GetPlaneName = "Outside C"
            
            Case "Wall-C"
            
                GetPlaneName = "Outside D"
            
            Case "Wall-D"
            
                GetPlaneName = "Outside A"

        End Select
        
    End If
           
End Function


Private Sub AddOverallDimension(oSubAssy As ISubAssy, swDrawing As SldWorks.DrawingDoc, swView As SldWorks.View, Clearance As Double)

    Dim swDisplayDim As SldWorks.DisplayDimension
    Set swDisplayDim = SelectAndAddDimension(oSubAssy.StartEdge, oSubAssy.EndEdge, swDrawing, _
                oSubAssy.EndComp.xMin - 0.01, oSubAssy.EndComp.yMin - Clearance, swView)
    Set oSubAssy.Dimension = swDisplayDim
    
End Sub

Private Function GetSubAssyComponentsIndexSorted(vComps As Variant, CompNoDict As Scripting.Dictionary)
    
    Dim vIdxArr As IArrList
    Set vIdxArr = New IArrList
    
    Dim i As Integer
    For i = LBound(vComps) To UBound(vComps)
    
        vIdxArr.AddtoList CompNoDict(vComps(i).Name2)
        
    Next i
    
    vIdxArr.SortItems False
    GetSubAssyComponentsIndexSorted = vIdxArr.Items

End Function

Private Function AddSplitLines(vCompsIdx As Variant, swDrawing As SldWorks.DrawingDoc, swView As SldWorks.View, _
        compDict As Scripting.Dictionary, CompNoDict As Scripting.Dictionary, IsFrontView As Boolean, _
        ByVal swLeftEdge As SldWorks.Edge, ByVal swRightEdge As SldWorks.Edge, Optional VisibleEdgesOnly As Boolean = True) As IArrListObject

    swDrawing.ActivateView swView.Name
    
    Dim vOutline As Variant
    vOutline = swView.GetOutline
    
    Dim i As Integer
    Dim NextAssyStartEdge As SldWorks.Edge
    
    Dim subAssylist As IArrListObject
    Set subAssylist = New IArrListObject
    
    Dim swLayerMgr As SldWorks.LayerMgr
    Set swLayerMgr = swDrawing.GetLayerManager
    
    Const LayName As String = "SPLIT LINE"
    
    Call CheckandAddLayer(LayName, "ASSEMBLY SPLIT LINE", swLayerMgr)

    For i = LBound(vCompsIdx) To UBound(vCompsIdx)
    
        Dim xMin As Double
        Dim yMin As Double
        Dim xMax As Double
        Dim yMax As Double

        Dim oComp As IComp
        Set oComp = compDict.Items(vCompsIdx(i))
            
        Call GetViewMaxMinPoints(oComp, swView, xMin, xMax, yMin, yMax)
        
        Dim swSketchSegment As SldWorks.SketchSegment
        Set swSketchSegment = swSketchMgr.CreateLine(xMax, yMax + 16 * 0.0254, _
                                0, xMax, yMin - 16 * 0.0254, 0)
                                
        'swSketchSegment.ConstructionGeometry = True
        swSketchSegment.Layer = LayName
        
        Call AddSplitLineNote(swSketchSegment, swDrawing, swView, "SPLIT LINE", False, 0.02)
        
        Dim swEdge As SldWorks.Edge
        Set swEdge = GetEdgeInView(oComp, swView, False, True, VisibleEdgesOnly)
        
        Call AddCollinearRelation(swDrawing, swEdge, swSketchSegment, swView)
        
        Dim oSubAssy As ISubAssy
        Dim swDisplayDim As SldWorks.DisplayDimension
    
        If IsFrontView Then
            
            If i = LBound(vCompsIdx) Then
            
                Set oSubAssy = New ISubAssy
                'Set swDisplayDim = SelectAndAddDimension(swLeftEdge, swEdge, swDrawing, _
                                oComp.xMin - 0.01, vOutline(1) - 0.015, swView)
                             
                Set oSubAssy.StartComp = compDict.Items(0)
                Set oSubAssy.EndComp = compDict.Items(vCompsIdx(i))
                
                Set oSubAssy.StartEdge = swLeftEdge
                Set oSubAssy.EndEdge = swEdge
                'Set oSubAssy.Dimension = swDisplayDim
                'oSubAssy.AssyLength = swDisplayDim.GetDimension2(0).Value
                oSubAssy.StartIdx = 0
                oSubAssy.EndIdx = vCompsIdx(i)
                
                subAssylist.AddtoList oSubAssy
                
            Else
            
                Set oSubAssy = New ISubAssy
                'Set swDisplayDim = SelectAndAddDimension(NextAssyStartEdge, swEdge, swDrawing, _
                                oComp.xMin - 0.01, vOutline(1) - 0.015, swView)
                                
                Set oSubAssy.StartComp = compDict.Items(vCompsIdx(i - 1) + 1)
                Set oSubAssy.EndComp = compDict.Items(vCompsIdx(i))
                                
                Set oSubAssy.StartEdge = NextAssyStartEdge
                Set oSubAssy.EndEdge = swEdge
                'Set oSubAssy.Dimension = swDisplayDim
                'oSubAssy.AssyLength = swDisplayDim.GetDimension2(0).Value
                oSubAssy.StartIdx = vCompsIdx(i - 1) + 1
                oSubAssy.EndIdx = vCompsIdx(i)
                
                subAssylist.AddtoList oSubAssy

            End If
            
            Dim NextAssyComp As IComp
            Set NextAssyComp = compDict.Items(vCompsIdx(i) + 1)
            
            Set NextAssyStartEdge = GetEdgeInView(NextAssyComp, swView, False, False, False)
            
            If i = UBound(vCompsIdx) Then
            
                Set oSubAssy = New ISubAssy
                'Set swDisplayDim = SelectAndAddDimension(swRightEdge, NextAssyStartEdge, swDrawing, _
                            NextAssyComp.xMax + 0.01, vOutline(1) - 0.015, swView)
                            
                Set oSubAssy.StartComp = compDict.Items(vCompsIdx(i) + 1)
                Set oSubAssy.EndComp = compDict.Items(UBound(compDict.Items))
                
                Set oSubAssy.StartEdge = NextAssyStartEdge
                Set oSubAssy.EndEdge = swRightEdge
                'Set oSubAssy.Dimension = swDisplayDim
                'oSubAssy.AssyLength = swDisplayDim.GetDimension2(0).Value
                
                oSubAssy.StartIdx = vCompsIdx(i) + 1
                oSubAssy.EndIdx = (CompNoDict.Count) - 1
                
                subAssylist.AddtoList oSubAssy
                            
            End If
            
        Else
            
            Dim TempComp As IComp
            Dim vPoint(2) As Double
            Dim vSheetPoint As Variant
    
            If i = LBound(vCompsIdx) Then
                
                Set TempComp = compDict.Items(0)
                Call GetViewMaxMinPoints(TempComp, swView, xMin, xMax, yMin, yMax)
                vPoint(0) = xMin
                vPoint(1) = yMin
                vPoint(2) = 0
                
                vSheetPoint = GetSketchPointInSheetSpace(swView, vPoint)
                
                Set swLeftEdge = GetEdgeInView(TempComp, swView, False, False)
                
                Set oSubAssy = New ISubAssy
                Set swDisplayDim = SelectAndAddDimension(swLeftEdge, swEdge, swDrawing, _
                                oComp.xMin - 0.01, vSheetPoint(1) - 0.005, swView, IsParanthesis:=True)
                                
                Set oSubAssy.StartEdge = swLeftEdge
                Set oSubAssy.EndEdge = swEdge
                Set oSubAssy.Dimension = swDisplayDim
                
                subAssylist.AddtoList oSubAssy
                
            Else
            
                Set oSubAssy = New ISubAssy
                Set swDisplayDim = SelectAndAddDimension(subAssylist.Items(UBound(subAssylist.Items)).EndEdge, swEdge, swDrawing, _
                                oComp.xMin - 0.01, vSheetPoint(1) - 0.005, swView, IsParanthesis:=True)
                                
                Set oSubAssy.StartEdge = subAssylist.Items(UBound(subAssylist.Items)).EndEdge
                Set oSubAssy.EndEdge = swEdge
                Set oSubAssy.Dimension = swDisplayDim
                
                subAssylist.AddtoList oSubAssy

            End If
            
            
            If i = UBound(vCompsIdx) Then
            
                Set TempComp = compDict.Items(UBound(compDict.Items))
                Set swRightEdge = GetEdgeInView(TempComp, swView, False, True)
            
                Set oSubAssy = New ISubAssy
                Set swDisplayDim = SelectAndAddDimension(swEdge, swRightEdge, swDrawing, _
                            oComp.xMax + 0.01, vSheetPoint(1) - 0.005, swView, IsParanthesis:=True)
                            
                Set oSubAssy.StartEdge = swEdge
                Set oSubAssy.EndEdge = swRightEdge
                Set oSubAssy.Dimension = swDisplayDim
                
                subAssylist.AddtoList oSubAssy
                            
            End If
            
        End If
        
    Next i
    
    Set AddSplitLines = subAssylist
    
End Function

Private Function GetControlSketch() As SldWorks.Component2

    Dim swTopLevelAssy As SldWorks.AssemblyDoc
    Set swTopLevelAssy = swTopLevelModel
    
    Dim vComps As Variant
    vComps = swTopLevelAssy.GetComponents(True)
    
    Dim i As Integer
    For i = LBound(vComps) To UBound(vComps)
    
        Dim swComp As SldWorks.Component2
        Set swComp = vComps(i)
        
        If InStr(swComp.Name2, "CONTROL") > 0 And InStr(swComp.Name2, "SKETCH") > 0 Then
            
            Dim vBodies As Variant
            Dim vBodiesInfo As Variant
            vBodies = swComp.GetBodies3(swBodyType_e.swSolidBody, vBodiesInfo)
            
            If IsEmpty(vBodies) Then
            
                Set GetControlSketch = swComp
                Exit Function
                
            End If
                
            
        End If

    Next i
       
End Function

Private Sub AddSplitLineNote(swSketchSegment As SldWorks.SketchLine, swDrawing As SldWorks.DrawingDoc, swView As SldWorks.View, _
            NoteText As String, Optional IsRight As Boolean = True, Optional ClearanceVal As Double = 0.005)

    
    Dim vPointInSheet As Variant
    vPointInSheet = SelectSketchSegment(swSketchSegment, swDrawing, swView, False, False)

    If IsRight Then
    
        Call AddNoteToView(swDrawing, NoteText, vPointInSheet(0) + ClearanceVal, vPointInSheet(1) + 0.00625)
        
    Else
        
        Call AddNoteToView(swDrawing, NoteText, vPointInSheet(0) - ClearanceVal, vPointInSheet(1) + 0.00625)
        
    End If
    
End Sub

Private Function GetCompDictionary(FlatCompList As Variant, CompNoDict As Scripting.Dictionary) As Scripting.Dictionary
    
    Dim TempDict As New Scripting.Dictionary
    
    Dim i As Integer
    For i = LBound(FlatCompList) To UBound(FlatCompList)
        
        TempDict.Add FlatCompList(i).GetComponent.Name2, FlatCompList(i)
        CompNoDict.Add FlatCompList(i).GetComponent.Name2, i
    
    Next i
    
    Set GetCompDictionary = TempDict
    
End Function

Function GetSelectedComponents() As Variant
    
    Dim swSelectionMgr As SldWorks.SelectionMgr
    Set swSelectionMgr = swTopLevelModel.SelectionManager
    
    Dim compDict As Scripting.Dictionary
    Set compDict = New Scripting.Dictionary
    
    If swSelectionMgr.GetSelectedObjectCount2(-1) > 0 Then
    
        Dim i As Integer
        For i = 0 To swSelectionMgr.GetSelectedObjectCount2(-1) - 1
            
            Dim swComp As SldWorks.Component2
            Set swComp = swSelectionMgr.GetSelectedObjectsComponent4(i + 1, -1)
            
            If False = compDict.Exists(swComp.Name2) Then
                
                compDict.Add swComp.Name2, swComp
            
            End If

        Next i
        
    End If
    
    If Not (compDict.Count = 0) Then
    
        GetSelectedComponents = compDict.Items
        
    End If

End Function
Private Sub ActivateDrawingDocument(swModel As SldWorks.ModelDoc2)
    
    Dim swFrame As SldWorks.Frame
    Set swFrame = swApp.Frame
    
    Dim vModelWindows As Variant
    vModelWindows = swFrame.ModelWindows
    
    Dim i As Integer
    For i = LBound(vModelWindows) To UBound(vModelWindows)
    
        Dim swModelWindow As SldWorks.ModelWindow
        Set swModelWindow = vModelWindows(i)
        
        If swModelWindow.Title = swModel.GetTitle Then
        
            swModelWindow.Activate
            Exit Sub
            
        End If
    
    Next i
End Sub

Private Function CheckForMultipleAssembly(ViewWidth As Double, ViewHeight As Double) As Boolean

    CheckForMultipleAssembly = False
    
    If ViewHeight <= 165 * 0.0254 Then
    
        If ViewWidth > 450 * 0.0254 Then
    
            CheckForMultipleAssembly = True
            
        End If
        
    Else
        
        If ViewWidth > 165 * 0.0254 Then
            
            CheckForMultipleAssembly = True
            
        End If
        
    End If

End Function

Private Function AddDimensionInFrontView(swView As SldWorks.View, FlatCompList As Variant, _
            MaxCompHeight As IComp, swDrawing As SldWorks.ModelDoc2, _
            ByRef swLeftEdge As SldWorks.Edge, ByRef swRightEdge As SldWorks.Edge) As SldWorks.Edge
            
    Dim vOutline As Variant
    vOutline = swView.GetOutline

    Dim LeftComp As IComp
    Set LeftComp = FlatCompList(0)
    
    Dim RightComp As IComp
    Set RightComp = FlatCompList(UBound(FlatCompList))
    
    Dim ClearanceLeft As Double
    Dim ClearanceRight As Double
    
    ClearanceLeft = GetClearance(LeftComp)
    ClearanceRight = GetClearance(RightComp)

    Set swLeftEdge = GetEdgeInView(LeftComp, swView, False, False)
    Set swRightEdge = GetEdgeInView(RightComp, swView, False, True)
            
    Dim swTopRightEdge As SldWorks.Edge
    Set swTopRightEdge = GetEdgeInView(RightComp, swView, True, True)
    
    Dim swBottomRightEdge As SldWorks.Edge
    Set swBottomRightEdge = GetEdgeInView(RightComp, swView, True, False)
    
    Dim swRightDim As SldWorks.DisplayDimension

    If (Abs(LeftComp.yMax - RightComp.yMax) <= 0.5 * 0.0254 * swView.ScaleDecimal) Then
    
        Dim MaxCompEdge As SldWorks.Edge
        Set MaxCompEdge = GetEdgeInView(MaxCompHeight, swView, True, True)
        
        Set swRightDim = SelectAndAddDimension(MaxCompEdge, _
                        swBottomRightEdge, swDrawing, RightComp.xMax + ClearanceRight, (vOutline(1) + vOutline(3)) / 2, swView)
    Else
    
        Dim swBottomLeftEdge As SldWorks.Edge
        Set swBottomLeftEdge = GetEdgeInView(LeftComp, swView, True, False)
        
        Dim swTopLeftEdge As SldWorks.Edge
        Set swTopLeftEdge = GetEdgeInView(LeftComp, swView, True, True)
        
        Set swRightDim = SelectAndAddDimension(swTopRightEdge, _
                        swBottomRightEdge, swDrawing, RightComp.xMax + ClearanceRight, (vOutline(1) + vOutline(3)) / 2, swView)
                        
        Dim swLeftDim As SldWorks.DisplayDimension
        Set swLeftDim = SelectAndAddDimension(swTopLeftEdge, _
            swBottomLeftEdge, swDrawing, LeftComp.xMin - ClearanceLeft, (vOutline(1) + vOutline(3)) / 2, swView)
        
    End If
    
    Set AddDimensionInFrontView = swBottomRightEdge

End Function

Private Function GetClearance(oComp As IComp) As Double

    If InStr(oComp.GetCustomProperty("Profile"), "CORNER") > 0 Then
        
        GetClearance = 0.01
        
    Else
        
        GetClearance = 0.02
        
    End If
        
End Function

Private Function SelectAndAddDimension(swEdge1 As SldWorks.Edge, swEdge2 As SldWorks.Edge, swDrawing As SldWorks.ModelDoc2, _
            xPos As Double, yPos As Double, swView As SldWorks.View, Optional IsDual As Boolean = True, Optional IsParanthesis As Boolean = False) As SldWorks.DisplayDimension
    
    If Not (swEdge1 Is Nothing) And Not (swEdge2 Is Nothing) Then
        
        swDrawing.ClearSelection2 True
        Call SelectEntity(swEdge1, False, swView)
        Call SelectEntity(swEdge2, True, swView)
        
        Set SelectAndAddDimension = swDrawing.AddHorizontalDimension2(xPos, yPos, 0)
        
        If Not SelectAndAddDimension Is Nothing Then
        
            SelectAndAddDimension.CenterText = True
            
            If IsDual Then
            
                SelectAndAddDimension.SetDual2 False, False
                
            End If
            
            If IsParanthesis Then
            
                SelectAndAddDimension.ShowParenthesis = True
                
            End If
            
        End If
    
    End If

End Function

Private Sub GetViewMaxMinPoints(oComp As IComp, swView As SldWorks.View, ByRef xMin As Double, _
                ByRef xMax As Double, ByRef yMin As Double, ByRef yMax As Double)

    Dim vViewMaxPt As Variant
    vViewMaxPt = GetComponentPointInViewSpace(oComp.GetComponent, oComp.GetMaxPointInModel, swView)
            
    Dim vViewMinPt As Variant
    vViewMinPt = GetComponentPointInViewSpace(oComp.GetComponent, oComp.GetMinPointInModel, swView)
    
    Call StrucutralElevationInsulation.GetMaxMinPoint(vViewMinPt(0), vViewMaxPt(0), xMin, xMax)
    Call StrucutralElevationInsulation.GetMaxMinPoint(vViewMinPt(1), vViewMaxPt(1), yMin, yMax)
    
End Sub
 
Private Function AddStructuralNotes(swDrawing As SldWorks.DrawingDoc, swSheet As SldWorks.Sheet, Is12GAPanelExists As Boolean, _
            IsAllPanels12GA As Boolean, IsDoorExists As Boolean, ByRef NoteCount As Integer, wallName As String) As SldWorks.Note

    swDrawing.ActivateSheet swSheet.GetName
    
    Dim swStructuralNote As SldWorks.Note
    Dim Note As String
    
    If Is12GAPanelExists Then
    
        NoteCount = 2
        If IsAllPanels12GA Then
        
            Note = "<FONT size=10PTS style=B>NOTES:" & vbCrLf & _
                "<FONT size=8PTS style=R>1. ALL PANELS ARE 12GA." & vbCrLf & _
             "2. RIB TO RIB #14 TEK SCREW @12" & Chr(34) & " O.C., UNLESS OTHERWISE SPECIFIED."
        
        Else
            Note = "<FONT size=10PTS style=B>NOTES:" & vbCrLf & _
                "<FONT size=8PTS style=R>1. ALL CIRCLED PANELS ARE 12GA." & vbCrLf & _
             "2. RIB TO RIB #14 TEK SCREW @12" & Chr(34) & " O.C., UNLESS OTHERWISE SPECIFIED."
             
        End If

    Else
    
        NoteCount = 1
        Note = "<FONT size=10PTS style=B> NOTES:" & vbCrLf & _
            "<FONT size=8PTS style=R>1. RIB TO RIB #14 TEK SCREW @12" & Chr(34) & " O.C., UNLESS OTHERWISE SPECIFIED."
         
     End If
     
    
    If InStr(wallName, "Wall") > 0 Then

        If IsDoorExists Then
     
            NoteCount = NoteCount + 1
            Note = Note & vbCrLf & NoteCount & ". DIMENSION FROM BOTTOM OF WALL PANEL TO BOTTOM HORIZONTAL FACE OF DOOR C-CHANNEL."
        
        End If
        
        NoteCount = NoteCount + 1
        Note = Note & vbCrLf & NoteCount & ". DIMENSION FROM BOTTOM OF WALL PANEL TO BOTTOM OF CEILING PANELS, USE FOR CEILING L-ANGLE PLACEMENT."
        
    End If
     
    Set swStructuralNote = swDrawing.CreateText2(Note, 1.99241243641486E-02, 6.92464210842187E-02, 0, 0, 0)
    swStructuralNote.SetTextJustification swTextJustification_e.swTextJustificationLeft
End Function

Private Sub InsertSketchBlock(swDrawing As SldWorks.DrawingDoc, swSheet As SldWorks.Sheet, ProjectNo As String)
    
    swDrawing.ActivateSheet swSheet.GetName
    
    Dim vSheetProp As Variant
    vSheetProp = swSheet.GetProperties
    
    Dim vPt(2) As Double
    vPt(0) = 0.01590679 * vSheetProp(3)
    vPt(1) = 0.00995866 * vSheetProp(3)
    vPt(2) = 0
    
    Dim SketchBlockInsertionPt As SldWorks.MathPoint
    Set SketchBlockInsertionPt = swMathUtility.CreatePoint(vPt)
    
    Dim swBlockDefinition As SldWorks.SketchBlockDefinition
    Set swBlockDefinition = swDrawing.SketchManager.MakeSketchBlockFromFile(SketchBlockInsertionPt, _
                "C:\FBD\COMMON\BLOCKS\" & ProjectNo & " INTERNAL ELEVATION KEY.SLDBLK", True, 1, 0)
                

End Sub

Private Sub UpdateBottomViewPosition(vComps As Variant, swDrawing As SldWorks.DrawingDoc, swView As SldWorks.View)
    
    Dim oComp As IComp
    Set oComp = vComps(0)
    
    Dim viewPosition As Variant
    viewPosition = swView.Position
    
    Dim wallPanelPosition As Variant
    wallPanelPosition = GetComponentPointInSheetSpace(oComp.GetComponent, oComp.GetMaxPointInModel, swView)
    
    If Abs(wallPanelPosition(1) - viewPosition(1)) / swView.ScaleDecimal > 5 * 0.0254 Then
    
        viewPosition(1) = viewPosition(1) + viewPosition(1) - wallPanelPosition(1)
        
    End If
    
    swView.Position = viewPosition
    
End Sub

Private Sub CleanUpActivateAndAddViewLabel(swDrawing As SldWorks.ModelDoc2, swView As SldWorks.View, wallName As String, _
        yPos As Double, Optional xPos As Double = 0, Optional InsulationName As String = "")

    swDrawing.SetUserPreferenceToggle swUserPreferenceToggle_e.swDisplayOrigins, False
    swDrawing.SetUserPreferenceToggle swUserPreferenceToggle_e.swDisplayPlanes, False
    swDrawing.SetUserPreferenceToggle swUserPreferenceToggle_e.swDisplayReferencePoints2, False
    swDrawing.SetUserPreferenceToggle swUserPreferenceToggle_e.swDisplayCurves, False
    swDrawing.SetUserPreferenceToggle swUserPreferenceToggle_e.swDisplayAllAnnotations, False
    swDrawing.SetUserPreferenceToggle swUserPreferenceToggle_e.swDisplayCompAnnotations, False
    swDrawing.SetUserPreferenceToggle swUserPreferenceToggle_e.swDisplayReferencePoints, False
    swDrawing.SetUserPreferenceToggle swUserPreferenceToggle_e.swDisplayLiveSections, False
    swDrawing.SetUserPreferenceToggle swUserPreferenceToggle_e.swDisplayLights, False
    
    swDrawing.ActivateView swView.Name
    
    Dim SheetDesc As String
    Dim LabelText As String
    If InStr(wallName, "Wall") > 0 Then
    
        If InsulationName = "" Then
    
            SheetDesc = "STRUCTURAL, ELEVATION, INSULATION, " & UCase(wallName)
            
        Else
        
            SheetDesc = "STRUCTURAL, ELEVATION, " & UCase(InsulationName) & ", " & UCase(wallName)
            
        End If
       
         
    Else
        
        SheetDesc = "STRUCTURAL, " & UCase(wallName) & ", " & UCase(InsulationName)
        
    End If
    
    LabelText = "<FONT size=10PTS style=B> $PRP:" & Chr(34) & "SHEET DESCRIPTION" & Chr(34)
    swDrawing.Extension.CustomPropertyManager("").Set2 "SHEET DESCRIPTION", SheetDesc
    swDrawing.Extension.CustomPropertyManager("").Set2 "ISSUED FOR", "CONSTRUCTION"
    
    If xPos = 0 Then
    
        Dim vOutline As Variant
        vOutline = swView.GetOutline
        xPos = (vOutline(0) + vOutline(2)) / 2
        
    End If
    
    Dim swLabelNote As SldWorks.Note

    Set swLabelNote = swDrawing.CreateText2(LabelText, xPos, yPos, 0, 0, 0)
    swLabelNote.SetTextJustification swTextJustification_e.swTextJustificationCenter
    
    swDrawing.Extension.Rebuild swRebuildOptions_e.swCurrentSheetDisp

End Sub


Private Sub AddCollinearRelation(swDrawing As SldWorks.DrawingDoc, swEdge As SldWorks.Edge, swSketchSegment As SldWorks.SketchSegment, swView As SldWorks.View)
    
    If Not (swEdge Is Nothing) And Not (swSketchSegment Is Nothing) Then
    
        Call SelectEntity(swEdge, False, swView)
        swSketchSegment.Select4 True, Nothing
                
        swDrawing.SketchAddConstraints "sgCOLINEAR"
        
    End If
    
End Sub

Function GetEdgeInView(oComp As IComp, swView As SldWorks.View, _
    IsHorizontal As Boolean, IsMax As Boolean, Optional CheckAllVisibleEdgesOnly As Boolean = True) As SldWorks.Edge
    
    
    Dim xMin As Double
    Dim yMin As Double
    Dim xMax As Double
    Dim yMax As Double
    Call GetViewMaxMinPoints(oComp, swView, xMin, xMax, yMin, yMax)
    
    Dim idx As Integer
    Dim ValToMatch As Double
    If IsHorizontal Then
        
        idx = 1
        If IsMax Then
        
            ValToMatch = yMax
            
        Else
        
             ValToMatch = yMin
             
        End If
        
    Else
    
        idx = 0
        
        If IsMax Then
        
            ValToMatch = xMax
            
        Else
        
             ValToMatch = xMin
             
        End If
        
    End If
    
    Dim swComp As SldWorks.Component2
    Set swComp = oComp.GetComponent
    

     Dim TempLength As Double
     TempLength = 0
        

    Dim vEnts As Variant
    If CheckAllVisibleEdgesOnly Then
    
        vEnts = swView.GetVisibleEntities2(swComp, swViewEntityType_e.swViewEntityType_Edge)
        
    Else
    
        vEnts = GetComponentEdges(swComp)
        
    End If

    If Not IsEmpty(vEnts) Then
    
        Dim i As Integer
        For i = LBound(vEnts) To UBound(vEnts)
        
            Dim swEdge As SldWorks.Edge
            Set swEdge = vEnts(i)
            
            Dim IsSelected As Boolean
            'IsSelected = SelectEntity(swEdge, False, swView)
            
            Dim swCurve As SldWorks.Curve
            Set swCurve = swEdge.GetCurve
            
            If swCurve.IsLine Then
            
                Dim vStartPoint As Variant
                vStartPoint = swEdge.GetStartVertex.GetPoint
                vStartPoint = GetComponentPointInViewSpace(swComp, vStartPoint, swView)
                
                Dim vEndPoint As Variant
                vEndPoint = swEdge.GetEndVertex.GetPoint
                vEndPoint = GetComponentPointInViewSpace(swComp, vEndPoint, swView)
                
                If Abs(vStartPoint(idx) - vEndPoint(idx)) <= 0.00001 And Abs(vStartPoint(idx) - ValToMatch) <= 0.00001 Then
                    
                    Dim vCurveParam As Variant
                    vCurveParam = swEdge.GetCurveParams2
                    
                    If swCurve.GetLength2(vCurveParam(6), vCurveParam(7)) > TempLength Then
                        
                        TempLength = swCurve.GetLength2(vCurveParam(6), vCurveParam(7))
                        Set GetEdgeInView = swEdge
                        
                    End If
                    
                End If
            
            End If
            
        Next i

    End If

End Function

Function GetEdgeInViewForBody(swComp As SldWorks.Component2, oBody As ISolidBody, swView As SldWorks.View, _
    IsHorizontal As Boolean, IsMax As Boolean, Optional CheckAllVisibleEdgesOnly As Boolean = True) As SldWorks.Edge
    
    
    Dim xMin As Double
    Dim yMin As Double
    Dim xMax As Double
    Dim yMax As Double
    
    Dim vPointMin(2) As Double
    vPointMin(0) = oBody.xMin
    vPointMin(1) = oBody.yMin
    vPointMin(2) = oBody.zMin
    
    Dim vPointMax(2) As Double
    vPointMax(0) = oBody.xMax
    vPointMax(1) = oBody.yMax
    vPointMax(2) = oBody.zMax
    
    Dim vViewPointMin As Variant
    vViewPointMin = GetSheetPointInViewSpace(swView, vPointMin)
   
    Dim vViewPointMax As Variant
    vViewPointMax = GetSheetPointInViewSpace(swView, vPointMax)
    
    Call StrucutralElevationInsulation.GetMaxMinPoint(vViewPointMin(0), vViewPointMax(0), xMin, xMax)
    Call StrucutralElevationInsulation.GetMaxMinPoint(vViewPointMin(1), vViewPointMax(1), yMin, yMax)
    
    Dim idx As Integer
    Dim ValToMatch As Double
    If IsHorizontal Then
        
        idx = 1
        If IsMax Then
        
            ValToMatch = yMax
            
        Else
        
             ValToMatch = yMin
             
        End If
        
    Else
    
        idx = 0
        
        If IsMax Then
        
            ValToMatch = xMax
            
        Else
        
             ValToMatch = xMin
             
        End If
        
    End If

     Dim TempLength As Double
     TempLength = 0
        

    Dim vEnts As Variant
    If CheckAllVisibleEdgesOnly Then
    
        vEnts = swView.GetVisibleEntities2(swComp, swViewEntityType_e.swViewEntityType_Edge)
        
    Else
    
        vEnts = oBody.GetBody.GetEdges
        
    End If

    If Not IsEmpty(vEnts) Then
    
        Dim i As Integer
        For i = LBound(vEnts) To UBound(vEnts)
        
            Dim swEdge As SldWorks.Edge
            Set swEdge = vEnts(i)

            Dim swCurve As SldWorks.Curve
            Set swCurve = swEdge.GetCurve
            
            If swCurve.IsLine Then
            
                Dim vStartPoint As Variant
                vStartPoint = swEdge.GetStartVertex.GetPoint
                vStartPoint = GetComponentPointInViewSpace(swComp, vStartPoint, swView)
                
                Dim vEndPoint As Variant
                vEndPoint = swEdge.GetEndVertex.GetPoint
                vEndPoint = GetComponentPointInViewSpace(swComp, vEndPoint, swView)
                
                If Abs(vStartPoint(idx) - vEndPoint(idx)) <= 0.00001 And Abs(vStartPoint(idx) - ValToMatch) <= 0.00001 Then
                    
                    Dim vCurveParam As Variant
                    vCurveParam = swEdge.GetCurveParams2
                    
                    If swCurve.GetLength2(vCurveParam(6), vCurveParam(7)) > TempLength Then
                        
                        TempLength = swCurve.GetLength2(vCurveParam(6), vCurveParam(7))
                        Set GetEdgeInViewForBody = swEdge
                        
                    End If
                    
                End If
            
            End If
            
        Next i

    End If

End Function


Function AddNoteToView(swDrawing As SldWorks.DrawingDoc, NoteText As String, xPos As Double, yPos As Double) As SldWorks.Annotation
            
    Dim swNote As SldWorks.Note
    Set swNote = swDrawing.InsertNote(NoteText)
    
    Dim swAnnotation As SldWorks.Annotation
            
    If Not swNote Is Nothing Then

        Set swAnnotation = swNote.GetAnnotation()

        If Not swAnnotation Is Nothing Then

            swAnnotation.SetPosition xPos, yPos, 0

        End If

    End If
    
    Set AddNoteToView = swAnnotation
    
End Function

Function SelectSketchSegment(swSketchSegment As SldWorks.SketchSegment, swDrawing As SldWorks.DrawingDoc, _
        swView As SldWorks.View, Append As Boolean, Optional IsNearEnd As Boolean = True, Optional PercentFromEnd As Double = 0.01)
    
    Dim swSketchLine As SldWorks.SketchLine
    Set swSketchLine = swSketchSegment
    
    Dim swStartPoint As SldWorks.sketchPoint
    Set swStartPoint = swSketchLine.GetStartPoint2
    
    Dim swEndPoint As SldWorks.sketchPoint
    Set swEndPoint = swSketchLine.GetEndPoint2

    Dim swCurve As SldWorks.Curve
    Set swCurve = swSketchSegment.GetCurve
        
    Dim LineLength As Double
    LineLength = swSketchSegment.GetLength
            
    Dim vLineParams As Variant
    vLineParams = swCurve.LineParams
        
    Dim vVectorData(2) As Double
    vVectorData(0) = vLineParams(3)
    vVectorData(1) = vLineParams(4)
    vVectorData(2) = vLineParams(5)
        
    Dim swMathVector As SldWorks.MathVector
    Set swMathVector = swMathUtility.CreateVector(vVectorData)
        
    Set swMathVector = swMathVector.Normalise
        
    Dim vSketchPoint(2) As Double
    Dim swMathPoint As SldWorks.MathPoint
        
    If IsNearEnd Then
            
        vSketchPoint(0) = swEndPoint.X
        vSketchPoint(1) = swEndPoint.Y
        vSketchPoint(2) = swEndPoint.Z
            
        Set swMathPoint = swMathUtility.CreatePoint(vSketchPoint)
        Set swMathVector = swMathVector.Scale(-1 * PercentFromEnd * LineLength)
            
    Else
 
        vSketchPoint(0) = swStartPoint.X
        vSketchPoint(1) = swStartPoint.Y
        vSketchPoint(2) = swStartPoint.Z

        Set swMathVector = swMathVector.Scale(PercentFromEnd * LineLength)
            
    End If
        
    Set swMathPoint = swMathUtility.CreatePoint(vSketchPoint)
    Set swMathPoint = swMathPoint.AddVector(swMathVector)
        
    vSketchPoint(0) = swMathPoint.ArrayData(0)
    vSketchPoint(1) = swMathPoint.ArrayData(1)
    vSketchPoint(2) = swMathPoint.ArrayData(2)
 
    Dim vPointInSheet As Variant
    vPointInSheet = StrucutralElevationInsulation.GetSketchPointInSheetSpace(swView, vSketchPoint)
    
    swDrawing.Extension.SelectByID2 "Line" & swSketchSegment.GetID(1), "SKETCHSEGMENT", vPointInSheet(0), vPointInSheet(1), vPointInSheet(2), Append, -1, Nothing, 0
    SelectSketchSegment = vPointInSheet
    
End Function

Private Sub AddCallouts(vBodies As Variant, swDrawing As SldWorks.ModelDoc2, swView As SldWorks.View, _
        yAssyMin As Double, yAssyMax As Double, IsUpper As Boolean, IsLower As Boolean, ByRef MaxClearnace As Double, Optional AddorSub As Integer = -1)
    
    Const SheetPosForLastBalloon As Double = 0.266
    Const Increment As Double = 0.005
    Dim MaxBalloonWidth As Double

    If Not IsEmpty(vBodies) Then
    
        swDrawing.Extension.SetUserPreferenceInteger swUserPreferenceIntegerValue_e.swDetailingBOMUpperText, swUserPreferenceOption_e.swDetailingNoOptionSpecified, swBalloonTextContent_e.swBalloonTextCutlistProperties
        
        Dim maxNoOfBalloons As Integer
        maxNoOfBalloons = 2
        
        MaxClearnace = maxNoOfBalloons * Increment
        
        Dim BalloonCount As Integer
        BalloonCount = maxNoOfBalloons
    
        Dim annXPos As Double
        Dim annYPos As Double
        
        Dim i As Integer
        For i = LBound(vBodies) To UBound(vBodies)
        
            Dim oBody As ISolidBody
            Set oBody = vBodies(i)
            
            swDrawing.ClearSelection2 True
    
            Dim xPos As Double
            Dim yPos As Double
          
            xPos = (oBody.xMin + oBody.xMax) / 2
    
            If Not (i = LBound(vBodies)) Then
        
                Dim prevBody As ISolidBody
                Set prevBody = vBodies(i - 1)
        
                If AddorSub = -1 Then
        
                    If Abs((prevBody.xMin + prevBody.xMax) / 2 - (oBody.xMin + oBody.xMax) / 2) > 2 * MaxBalloonWidth Then
        
                        AddorSub = 1
                        BalloonCount = 1
                        
                    ElseIf Abs((prevBody.xMin + prevBody.xMax) / 2 - (oBody.xMin + oBody.xMax) / 2) > MaxBalloonWidth Then
                    
                        AddorSub = 1
                        
                        If BalloonCount = 0 Then

                            BalloonCount = maxNoOfBalloons
                            
                        ElseIf BalloonCount >= maxNoOfBalloons Then
                        
                            BalloonCount = 1
                            
                        End If
        
                    End If
        
                Else
        
                    If Abs((prevBody.xMin + prevBody.xMax) / 2 - (oBody.xMin + oBody.xMax) / 2) > MaxBalloonWidth Then
        
                        AddorSub = 1
                        BalloonCount = 1

                    End If
        
                End If
                
                If i = UBound(vBodies) Then
                
                    AddorSub = 1
                    
                End If
                
                If AddorSub = 1 Then
                
                    If BalloonCount > maxNoOfBalloons Then

                        If Not (i = UBound(vBodies)) Then
                        
                            AddorSub = -1
                            BalloonCount = BalloonCount + AddorSub
                            
                        Else
                        
                            MaxClearnace = BalloonCount * Increment
                            
                        End If
                        
                    End If
                
                Else
                
                    If BalloonCount < 1 Then
                    
                        BalloonCount = maxNoOfBalloons
                        
                    End If
                    
                End If
                

                
            End If
     
            If IsUpper Then
                
                yPos = 0.9 * oBody.yMax + 0.1 * oBody.yMin
                annYPos = yAssyMax + BalloonCount * Increment
                
            End If
            
            If IsLower Then
            
                yPos = 0.1 * oBody.yMax + 0.9 * oBody.yMin
                annYPos = yAssyMin - BalloonCount * Increment
                
            End If
            
            If False = IsUpper And False = IsLower Then
                        
                yPos = 0.1 * oBody.yMax + 0.9 * oBody.yMin
                annYPos = oBody.yMin - BalloonCount * Increment
    
            End If
            
            annXPos = xPos
            BalloonCount = BalloonCount + AddorSub
    
            
    
            Dim IsSelected As Boolean
            IsSelected = SelectFaceWithPosition(swDrawing, oBody, xPos, yPos)
    
            If False = IsSelected Then
            
                Dim vFaces As Variant
                vFaces = swView.GetVisibleEntities2(oBody.GetComponent, swViewEntityType_e.swViewEntityType_Face)
                IsSelected = SelectFaceOfTheBody(vFaces, oBody, swDrawing, swView, False)
    
            End If
    
            If IsSelected Then
            
                Dim swBalloonParams As SldWorks.BalloonOptions
                Set swBalloonParams = swDrawing.Extension.CreateBalloonOptions()
                swBalloonParams.Size = swBalloonFit_e.swBF_Tightest
                swBalloonParams.Style = swBalloonStyle_e.swBS_Inspection
                swBalloonParams.UpperTextContent = swBalloonTextContent_e.swBalloonTextCutlistProperties
                swBalloonParams.UpperText = "$PRPWLD:" & Chr(34) & "PART CODE" & Chr(34) '"$PRPWLD:" & Chr(34) & "ITEM NO" & Chr(34)
                
                
                Dim swNote As SldWorks.Note
                Set swNote = swDrawing.Extension.InsertBOMBalloon2(swBalloonParams)
                
                If Not swNote Is Nothing Then

                    swNote.PropertyLinkedText = "$PRPWLD:" & Chr(34) & "PART CODE" & Chr(34)
                    
                    Dim swAnn As SldWorks.Annotation
                    Set swAnn = swNote.GetAnnotation
                    swAnn.SetPosition2 annXPos, annYPos, 0
                    
                    Dim HeadStyle As Integer
                    
                    swAnn.SetLeader3 swLeaderStyle_e.swAlwaysAttachToBalloon + swLeaderStyle_e.swSTRAIGHT, swLeaderSide_e.swLS_SMART, False, False, True, False
                    HeadStyle = swAnn.SetArrowHeadStyleAtIndex(0, swArrowStyle_e.swCLOSED_ARROWHEAD)
                    
                    Dim vNoteExtents As Variant
                    vNoteExtents = swNote.GetExtent
                    
                    MaxBalloonWidth = ((vNoteExtents(3) - vNoteExtents(0))) + 0.0027
    
                    
                    If AddorSub = 1 Then

                        annXPos = xPos - MaxBalloonWidth + 0.0054
    
                        swAnn.SetPosition2 annXPos, annYPos, 0
                        
                    End If
                    
                End If
                
            End If
    
        Next i
        
    End If


End Sub

Private Sub SelectBody(swDrawing As SldWorks.ModelDoc2, oBody As ISolidBody, xPos As Double, _
    yPos As Double, Count As Integer, IsSelected As Boolean, swView As SldWorks.View)
    
    IsSelected = swDrawing.Extension.SelectByID2("", "FACE", xPos, yPos, _
                    0, False, -1, Nothing, 1)
                    
    If Count > 2 Then
        
        Dim vFaces As Variant
        vFaces = oBody.GetViewNormalFaces
        
        Dim swFace As SldWorks.Face2
        Set swFace = vFaces(0)
        IsSelected = SelectEntity(swFace, False, swView)
        Exit Sub
        
    End If
    
    If IsSelected Then
    
        Dim swSelectMgr As SldWorks.SelectionMgr
        Set swSelectMgr = swDrawing.SelectionManager
        
        Dim swComp As SldWorks.DrawingComponent
        Set swComp = swSelectMgr.GetSelectedObjectsComponent4(2, -1)
        
        If Not (Right(swComp.Name, Len(swComp.Name) - InStrRev(swComp.Name, "/")) = _
            Right(oBody.GetComponent.Name2, Len(oBody.GetComponent.Name2) - InStrRev(oBody.GetComponent.Name2, "/"))) Then
            
            Call SelectComponent(swDrawing, oBody, (oBody.xMax + oBody.xMin) / 2, yPos, Count + 1, IsSelected, swView)
            
        End If
        
    Else
    
        Call SelectComponent(swDrawing, oBody, (oBody.xMax + oBody.xMin) / 2, yPos, Count + 1, IsSelected, swView)
        
    End If
    
    
End Sub

Function GetViewName(wallName As String)

    Select Case wallName
        
        Case "Wall-A"
            
            GetViewName = "*Back"
        
        Case "Wall-B"
            
            GetViewName = "*Right"
        
        Case "Wall-C"
        
            GetViewName = "*Front"
        
        Case "Wall-D"
            
            GetViewName = "*Left"
            
        Case "Ceiling"
            
            GetViewName = "*Top"
            
        Case Else
            
            ViewNameForm.Show
            GetViewName = ViewNameForm.ViewNameBox.Value
            Unload ViewNameForm
    
    End Select
    
End Function

Function ScaleView(swDrawing As SldWorks.DrawingDoc, swView As SldWorks.View, _
            ViewWidth As Double, ViewHeight As Double) As SldWorks.View
            

    Dim xScale As Integer
    Dim yScale As Integer
    xScale = GetScaleValue(ViewWidth / (swView.ScaleDecimal * 0.371))
    yScale = GetScaleValue(ViewHeight / (swView.ScaleDecimal * 0.1295)) '0.20995
    
    Dim IsScaleSet As Boolean
    IsScaleSet = False
    
    If xScale > 0 And yScale > 0 Then
        
        If yScale > xScale Then
            
            IsScaleSet = swView.Sheet.SetScale(1, yScale, True, True)
           
        Else
            
            IsScaleSet = swView.Sheet.SetScale(1, xScale, True, True)
        
        End If
        
    End If
    
    Dim IsViewSelected As Boolean
    
    Dim swDrawingModel As SldWorks.ModelDoc2
    Set swDrawingModel = swDrawing
    
'    IsViewSelected = swDrawingModel.Extension.SelectByID2(swView.Name, "DRAWINGVIEW", 0, 0, 0, False, 0, Nothing, 0)
'    Set ScaleAndInsertBottomView = swDrawing.CreateUnfoldedViewAt3(0.21593179, 0.065, 0, False)

End Function

Function GetScaleValue(scaleVal As Double) As Integer

    GetScaleValue = 0
    
    Dim stdScales As Variant
    stdScales = Array(1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 384)
    
    Dim i As Integer
    For i = LBound(stdScales) To UBound(stdScales)
    
        If stdScales(i) >= scaleVal Then
           GetScaleValue = stdScales(i)
           Exit For
        End If
    
    Next i

End Function

Function GetComponentsSortedWithXPosition(swView As SldWorks.View, swDrawing As SldWorks.ModelDoc2, _
       ProfileTextToMatch As String) As IArrListObject
    
    swDrawing.ActivateView swView.Name

    Dim vComps As Variant
    vComps = swView.GetVisibleComponents()

    Dim swTopLevelAssy As SldWorks.AssemblyDoc
    Set swTopLevelAssy = swTopLevelModel

    Dim CompList As IArrListObject
    Set CompList = New IArrListObject

    Dim i As Integer
    For i = LBound(vComps) To UBound(vComps)
    
        Dim swComp As SldWorks.Component2
        Set swComp = vComps(i)
        
        Dim swCompFromRoot As SldWorks.Component2
        Set swCompFromRoot = GetComponentFromRoot(swComp.Name2, swComp, swTopLevelAssy)
            
        If swCompFromRoot.GetSuppression = swComponentSuppressionState_e.swComponentLightweight Then
            
            Dim bRet As Integer
            bRet = swCompFromRoot.SetSuppression2(swComponentSuppressionState_e.swComponentResolved)
            
        End If

        Dim swCompModel As SldWorks.ModelDoc2
        Set swCompModel = swCompFromRoot.GetModelDoc2

        If Not swCompModel Is Nothing Then
            
            Dim swCompProp As SldWorks.CustomPropertyManager
            Set swCompProp = swCompModel.Extension.CustomPropertyManager("")
            
            Dim swBody As SldWorks.Body2
            Set swBody = swCompFromRoot.GetBody
            
            If Not swBody Is Nothing Then
                
                If swBody.IsSheetMetal Then
            
                    Dim Profile As String
                    Dim ResolvedVal As String
                    Dim wasResolved As Boolean
                    swCompProp.Get5 "Profile", False, Profile, ResolvedVal, wasResolved
                    
                    If InStr(Profile, ProfileTextToMatch) > 0 Then
                    
                        CompList.AddtoList GetComponentWithPosition(swCompFromRoot, swView, swDrawing)
                    
                    End If
                    
                End If
                
            End If
        
        End If
        
    Next i

    CompList.SortItems "xMin", False
        
    Set GetComponentsSortedWithXPosition = CompList

End Function

Sub GetComponentBoundsInView(CompList As IArrListObject, ByRef ViewWidth As Double, _
    ByRef ViewHeight As Double, MaxHeightComp As IComp)

    CompList.SortItems "yMin", False
    
    Dim MinHeight As Double
    MinHeight = CompList.Items(LBound(CompList.Items)).yMin
    
    CompList.SortItems "yMax"
    Set MaxHeightComp = CompList.Items(LBound(CompList.Items))
    
    ViewHeight = MaxHeightComp.yMax - MinHeight
    
    CompList.SortItems "yMax", True
    CompList.SortItems "xMin", False
    ViewWidth = CompList.Items(UBound(CompList.Items)).xMax - CompList.Items(LBound(CompList.Items)).xMin

    
End Sub

Function GetComponentFromRoot(AssyName As String, swComp As SldWorks.Component2, swTopLevelAssy As SldWorks.AssemblyDoc) As SldWorks.Component2

        Dim compName As String
        compName = Right(AssyName, Len(AssyName) - InStrRev(AssyName, "/"))

        Dim TempComp As SldWorks.Component2
        Set TempComp = swTopLevelAssy.GetComponentByName(compName)

        If Not InStr(AssyName, TempComp.Name2) > 0 Then

            AssyName = Replace(AssyName, "/" & compName, "")

            Dim swAssy As SldWorks.Component2
            Set swAssy = GetComponentFromRoot(AssyName, swComp, swTopLevelAssy)

            Set GetComponentFromRoot = GetThisChildrenOfAssy(swAssy, compName)
            
        Else
        
            Set GetComponentFromRoot = TempComp

            
        End If
        
       
        
End Function

Function GetThisChildrenOfAssy(swAssy As SldWorks.Component2, compName As String) As SldWorks.Component2

    Dim vChild As Variant
    vChild = swAssy.GetChildren
    
    Dim i As Integer
    For i = LBound(vChild) To UBound(vChild)
    
        Dim swComp As SldWorks.Component2
        Set swComp = vChild(i)
        
        Debug.Print swComp.Name2
        
        If InStr(swComp.Name2, compName) > 0 Then
        
            Set GetThisChildrenOfAssy = swComp
            Exit For
        
        End If
    
    Next i

End Function

Function GetComponentWithPosition(swComp As SldWorks.Component2, swView As SldWorks.View, _
        swDrawing As SldWorks.ModelDoc2) As IComp

    Dim MinPoint As Variant
    Dim MaxPoint As Variant
    Dim vBodyMinPoint(2) As Double
    Dim vBodyMaxPoint(2) As Double
    Call GetMinMaxBodyPointsInSheetSpace(swComp, MinPoint, MaxPoint, vBodyMinPoint, vBodyMaxPoint, swView)
            
    Dim oComp As IComp
    Set oComp = New IComp
    oComp.Initialize swComp, MinPoint, MaxPoint, vBodyMinPoint, vBodyMaxPoint

    Set GetComponentWithPosition = oComp

End Function

Private Sub GetMinMaxBodyPointsInSheetSpace(swComp As SldWorks.Component2, _
        ByRef MinPoint As Variant, ByRef MaxPoint As Variant, ByRef vBodyMinPoint() As Double, _
            ByRef vBodyMaxPoint() As Double, swView As SldWorks.View, Optional IsCorZ As Boolean = False)
            
    Dim vBodies As Variant

    If IsCorZ Then

        Dim vBodyInfo As Variant
        vBodies = swComp.GetBodies3(swBodyType_e.swSolidBody, vBodyInfo)

    Else

        vBodies = swComp.GetModelDoc2.GetBodies(swSolidBody)
        
    End If
    
    Dim swBody As SldWorks.Body2
    Set swBody = vBodies(0)

    Dim vBodyBounds As Variant
    vBodyBounds = swBody.GetBodyBox

    vBodyMinPoint(0) = vBodyBounds(0)
    vBodyMinPoint(1) = vBodyBounds(1)
    vBodyMinPoint(2) = vBodyBounds(2)
            
    vBodyMaxPoint(0) = vBodyBounds(3)
    vBodyMaxPoint(1) = vBodyBounds(4)
    vBodyMaxPoint(2) = vBodyBounds(5)

            
    MinPoint = GetComponentPointInSheetSpace(swComp, vBodyMinPoint, swView)
    MaxPoint = GetComponentPointInSheetSpace(swComp, vBodyMaxPoint, swView)
    
End Sub

Function GetComponentEdges(swComp As SldWorks.Component2)
    
    Dim TempEdges As Variant
    
    Dim vBodies As Variant
    vBodies = swComp.GetBodies3(swBodyType_e.swSolidBody, swBodyInfo_e.swNormalBody_e)
    
    Dim i As Integer
    Dim j As Integer
    For i = LBound(vBodies) To UBound(vBodies)
    
        Dim swBody As SldWorks.Body2
        Set swBody = vBodies(i)
        
        Dim vEdges As Variant
        vEdges = swBody.GetEdges
        
        If i = 0 Then
            
            TempEdges = vEdges
            
        Else
            
            TempEdges = CombineArr(TempEdges, vEdges)
            
        End If
    
    Next i
    
    GetComponentEdges = TempEdges

End Function

Function CombineArr(ByVal MainArr As Variant, ArrToAdd As Variant)

    Dim i As Integer
    For i = LBound(ArrToAdd) To UBound(ArrToAdd)
    
        ReDim Preserve MainArr(UBound(MainArr) + 1)
        Set MainArr(UBound(MainArr)) = ArrToAdd(i)
        
    Next i
    
    CombineArr = MainArr
    
End Function




