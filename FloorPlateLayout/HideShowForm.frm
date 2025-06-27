VERSION 5.00
Begin {C62A69F0-16DC-11CE-9E98-00AA00574A4F} HideShowForm 
   Caption         =   "Hide/ Show Components"
   ClientHeight    =   4572
   ClientLeft      =   108
   ClientTop       =   456
   ClientWidth     =   6636
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
Const BalloonWidth As Double = 0.0065
Const SheetBorderTop As Double = 0.27030866
Const SheetBorderLeft As Double = 0.01590679
Const SheetBorderRight As Double = 0.41595679
Dim compDict As Scripting.Dictionary
Const HorizontalMaxDim As Double = 0.371
Const VerticalMaxDim As Double = 0.1295

Private Sub AddCompButton_Click()

    Dim swModel As SldWorks.ModelDoc2
    Set swModel = swApp.ActiveDoc
    
    Dim swSelect As SldWorks.SelectionMgr
    Set swSelect = swModel.SelectionManager
    
    If swSelect.GetSelectedObjectCount2(-1) > 0 Then
    
        Dim i As Integer
        For i = 1 To swSelect.GetSelectedObjectCount2(-1)
        
            If swSelect.GetSelectedObjectType3(i, -1) = swSelectType_e.swSelFTRFOLDER Then
                
                Dim swFeat As SldWorks.Feature
                Set swFeat = swSelect.GetSelectedObject6(i, -1)
                
                Dim vCompArr() As SldWorks.Component2
                ReDim vCompArr(0)
                
                Call GetComponentsFromFolder(swFeat, vCompArr, 0)
                Call AddComponentsToDictionary(vCompArr)

            Else
        
                Dim swComp As SldWorks.Component2
                Set swComp = swSelect.GetSelectedObjectsComponent4(i, -1)
                Call CheckAndAddToList(swComp)
                
            End If

        Next i

    Else

        Me.CompListButton.BackColor = vbRed
        MsgBox "No Components were selected"
    
    End If
    
    swModel.ClearSelection2 True
    
End Sub

Sub AddComponentsToDictionary(CompArr As Variant)

    Dim i As Integer
    For i = LBound(CompArr) To UBound(CompArr)
    
        Dim swComp As SldWorks.Component2
        Set swComp = CompArr(i)
        
        Call CheckAndAddToList(swComp)
    
    Next i

End Sub

Private Sub CheckAndAddToList(swComp As SldWorks.Component2)

    If Not (compDict.Exists(swComp.Name2)) Then

        Me.CompListButton.AddItem
        Me.CompListButton.List(Me.CompListButton.ListCount - 1, 0) = swComp.Name2
        compDict.Add swComp.Name2, swComp
            
    End If
    
End Sub

Private Sub clearListButton_Click()

    Dim i As Integer
    With Me.CompListButton

        For i = .ListCount - 1 To 0 Step -1
                         
            compDict.Remove .List(i, 0)
            .RemoveItem (i)
                    
        Next i
        
    End With

    
End Sub

Function GetComponentsFromFolder(swFeat As SldWorks.Feature, ByRef TempCompArr As Variant, Level As Integer)

    Dim swFeatFolder As SldWorks.FeatureFolder
    Set swFeatFolder = swFeat.GetSpecificFeature2
            
    Dim i As Integer
    Dim vFeats As Variant
    vFeats = swFeatFolder.GetFeatures
            
    For i = LBound(vFeats) To UBound(vFeats)
                
        Dim compFeat As SldWorks.Feature
        Set compFeat = vFeats(i)
        
        Debug.Print compFeat.Name
                
        If Not (compFeat.GetTypeName2 = "FtrFolder") Then
                
            Dim swComp As SldWorks.Component2
            Set swComp = compFeat.GetSpecificFeature2
                    
            If Not swComp Is Nothing Then
            
                Set TempCompArr(UBound(TempCompArr)) = swComp
                ReDim Preserve TempCompArr(UBound(TempCompArr) + 1)
                
            End If
                
        Else
                
            Call GetComponentsFromFolder(compFeat, TempCompArr, Level + 1)
                             
        End If

    Next i
    
    If Level = 0 Then
    
        ReDim Preserve TempCompArr(UBound(TempCompArr) - 1)
        
    End If
    
End Function
    
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

Private Sub UserForm_QueryClose(Cancel As Integer, CloseMode As Integer)

    Unload DrawingForm

End Sub


Private Sub CreateButton_Click()
 
    Me.Hide

    Dim ProjectNo As String
    ProjectNo = DrawingForm.ProjectNoBox.Value
    
    Dim WeldmentNo As String
    WeldmentNo = Me.WeldNoBox.Value
    
    Unload DrawingForm

    Set swMathUtility = swApp.GetMathUtility
    
    Call ActivateAndRebuildComponent(swFloorWeldment, False)
    Call swTopLevelModel.Extension.Rebuild(swRebuildOptions_e.swForceRebuildAll)

    Dim swDrawing As SldWorks.DrawingDoc
    Set swDrawing = swApp.NewDocument("C:\FBD\COMMON\FBD Templates\DEFAULT\METAL FAB DRAWING.DRWDOT", 0, 0, 0)

    Set swSketchMgr = swDrawing.SketchManager

    Dim swSheet As SldWorks.Sheet
    Set swSheet = swDrawing.GetCurrentSheet

    Call InsertSketchBlock(swDrawing, swSheet, ProjectNo)

    Dim swTopView As SldWorks.View
    Set swTopView = swDrawing.CreateDrawViewFromModelView3(swTopLevelModel.GetPathName(), "*Top", 0.21593179, 0.15578398, 0)
    
    Dim oFloorComp As IComp
    Set oFloorComp = New IComp
    
    oFloorComp.Initialize swFloorWeldment, swTopView
    
    Dim ViewWidth As Double
    ViewWidth = oFloorComp.xMax - oFloorComp.xMin
    
    Dim ViewHeight As Double
    ViewHeight = oFloorComp.yMax - oFloorComp.yMin
    
    Call RotateAndScaleView(swDrawing, swTopView, ViewWidth, ViewHeight)
    Call UpdateTopViewPosition(oFloorComp, swDrawing, swTopView)
    
    Dim FloorPlateList As IArrListObject
    Set FloorPlateList = GetFloorPlateList(compDict.Items, swTopView)
    
    Dim yMinFloorDict As Scripting.Dictionary
    Set yMinFloorDict = GetConsolidatedDict(FloorPlateList, "yMin", swTopView)
    
    Dim xMinFloorDict As Scripting.Dictionary
    Set xMinFloorDict = GetConsolidatedDict(FloorPlateList, "xMin", swTopView)

    Dim xMaxPlateList As IArrListObject
    Set xMaxPlateList = GetCompListBasedonLocationParam(FloorPlateList, "xMax", swTopView)
    
    Dim yMaxPlateList As IArrListObject
    Set yMaxPlateList = GetCompListBasedonLocationParam(FloorPlateList, "yMax", swTopView)
    
    Dim vBlockOutList As IArrListObject
    Set vBlockOutList = GetBlockOutList(FloorPlateList.Items, swTopView)
    
    Dim yMinBlockOutDict As Scripting.Dictionary
    Set yMinBlockOutDict = GetConsolidatedDict(vBlockOutList, "yMin", swTopView)
    
    Dim xMinBlockOutDict As Scripting.Dictionary
    Set xMinBlockOutDict = GetConsolidatedDict(vBlockOutList, "xMin", swTopView)
    
    Dim ClonedBlockOutList As IArrListObject
    Set ClonedBlockOutList = vBlockOutList.Clone
    
    Dim yMaxBlockOutDict As Scripting.Dictionary
    Set yMaxBlockOutDict = GetConsolidatedDict(ClonedBlockOutList, "yMax", swTopView)
    
    Dim xMaxBlockOutDict As Scripting.Dictionary
    Set xMaxBlockOutDict = GetConsolidatedDict(ClonedBlockOutList, "xMax", swTopView)

    
    Call FindAndAddBeforeBlockOuts(xMaxBlockOutDict, ClonedBlockOutList, "xMin", BlockOutSide_e.Left)
    Call FindAndAddAfterBlockOuts(xMinBlockOutDict, ClonedBlockOutList, "xMax", BlockOutSide_e.Right)

    Call FindAndAddBeforeBlockOuts(yMaxBlockOutDict, ClonedBlockOutList, "yMin", BlockOutSide_e.Bottom)
    Call FindAndAddAfterBlockOuts(yMinBlockOutDict, ClonedBlockOutList, "yMax", BlockOutSide_e.Top)
    
    Call SegregateAndAddDimensionVertically(xMinBlockOutDict, xMaxPlateList, xMinFloorDict, oFloorComp, swDrawing, swTopView)
    Call SegregateAndAddDimensionHorizontally(yMinBlockOutDict, yMaxPlateList, yMinFloorDict, oFloorComp, swDrawing, swTopView)

    
    
    swApp.SetUserPreferenceToggle swUserPreferenceToggle_e.swSketchInference, False
    Call AddCrossMarkAndBalloons(ClonedBlockOutList, swDrawing, swTopView, oFloorComp)
    
    Call vBlockOutList.SortItems("xMin", False)
    

    
    
    

    

    
    
    'Call AddOrdinateToHorizontalPerimeterBeams(BottomBeamList, swDrawing, swTopView, False)
    'Call AddOrdinateToHorizontalPerimeterBeams(TopBeamList, swDrawing, swTopView, True)
    
    'Call AddHorizontalAssyOrdinate(BottomBeamList.Items, TopBeamList.Items, swDrawing, swTopView)

    'Call AddSeeNote2Circle(swDrawing, swTopView, BottomBeamList.Items, RightBeamList.Items, False)
    
    swDrawing.ClearSelection2 True
    Call AddNoteToView(swDrawing, "<FONT size=10PTS style=B>TOP VIEW WITH FLOOR PLATES", _
        (oFloorComp.xMax + oFloorComp.xMin) / 2, oFloorComp.yMin - 0.025)
    
    
    'Call AddEllipseAndCreateDetailView(swDrawing, swTopView, VerticalSubWeldmentList, LegendAscii, IsAsciiMaxReached, SubWeldmentViewDict, SubWeldBodyDict, IsSubWeldmentExists)
    
    
    Call EditTemplate(swDrawing, swDrawing.GetCurrentSheet, WeldmentNo, "CUTLIST AND BEAM DETAILS")
 
    Call AddStructuralNotes(swDrawing)

    Call SetHiddenEdgesVisibleAndRemoveTangentEdges(swTopView, swDrawing)


    
    Call swDrawing.Extension.Rebuild(swRebuildOptions_e.swCurrentSheetDisp)
    

    Set oFloorComp = Nothing
    Set swFloorWeldment = Nothing

    swApp.SetUserPreferenceToggle swUserPreferenceToggle_e.swSketchInference, True
    
    Unload Me

End Sub


Sub ActivateAndRebuildComponent(swComp As Object, Optional ToClose As Boolean = True)
    
    Dim swDoc As SldWorks.ModelDoc2
    Set swDoc = swApp.ActivateDoc3(swComp.GetPathName, True, swRebuildOnActivation_e.swDontRebuildActiveDoc, Err)
    
    If Not swDoc Is Nothing Then
        
        Call swDoc.Extension.Rebuild(swRebuildOptions_e.swForceRebuildAll)
        
        If ToClose Then
            swApp.CloseDoc swDoc.GetPathName
        End If
        
    End If
    
End Sub

Function GetCompListBasedonLocationParam(ArrList As IArrListObject, Param As String, swView As SldWorks.View) As IArrListObject
    
    Set GetCompListBasedonLocationParam = New IArrListObject
    
    If ArrList.Count > 0 Then
        
        ArrList.SortItems Param
        
        Dim vItems As Variant
        vItems = ArrList.Items
        
        Dim keyVal As Double
        
        Dim i As Integer
        For i = LBound(vItems) To UBound(vItems)
        
            Dim oComp As IComp
            Set oComp = vItems(i)
            
            If i = LBound(vItems) Then
                
                keyVal = CallByName(oComp, Param, VbGet)
                GetCompListBasedonLocationParam.AddtoList oComp
                
            Else
            
                If Abs(keyVal - CallByName(oComp, Param, VbGet)) < 0.125 * 0.0254 * swView.ScaleDecimal Then
                
                    GetCompListBasedonLocationParam.AddtoList oComp
                
                End If
            
            End If
        
        Next i

    End If
    
End Function

Function GetConsolidatedDict(MainArrList As IArrListObject, Param As String, swView As SldWorks.View) As Scripting.Dictionary
    
    MainArrList.SortItems Param, False
    Set GetConsolidatedDict = New Scripting.Dictionary
    
    If MainArrList.Count > 0 Then
        
        Dim vComps As Variant
        vComps = MainArrList.Items
        
        Dim i As Integer
        For i = LBound(vComps) To UBound(vComps)
        
            Dim oComp As Object
            Set oComp = vComps(i)

            Dim keyVal As Double
            keyVal = CallByName(oComp, Param, VbGet)
            
            Dim ArrList As IArrListObject
            
            If GetConsolidatedDict.Exists(keyVal) Then

                Set ArrList = GetConsolidatedDict.Item(keyVal)
                ArrList.AddtoList oComp

            Else

                If i = LBound(vComps) Then
                
                    Set ArrList = New IArrListObject
                    ArrList.AddtoList oComp
                    GetConsolidatedDict.Add keyVal, ArrList
                       
                Else
                
                    Dim PrevKeyVal As Double
                    PrevKeyVal = GetConsolidatedDict.Keys(UBound(GetConsolidatedDict.Keys))
                    
                    If Abs(PrevKeyVal - CallByName(oComp, Param, VbGet)) < 0.125 * 0.0254 * swView.ScaleDecimal Then
                        
                       Set ArrList = GetConsolidatedDict.Item(PrevKeyVal)
                       ArrList.AddtoList oComp
                    
                    Else
                        
                        Set ArrList = New IArrListObject
                        ArrList.AddtoList oComp
                        GetConsolidatedDict.Add keyVal, ArrList
                    
                    End If
                
                End If
                
            End If
        
        Next i
        
    End If
    
End Function

Function GetFloorPlateList(vComps As Variant, swView As SldWorks.View) As IArrListObject

    Set GetFloorPlateList = New IArrListObject

    If Not IsEmpty(vComps) Then
        
        Dim i As Integer
        For i = LBound(vComps) To UBound(vComps)
        
            Dim swComp As SldWorks.Component2
            Set swComp = vComps(i)
            
            Dim oComp As IComp
            Set oComp = New IComp
            
            If False = swComp.IsSuppressed Then
            
                oComp.Initialize swComp, swView
                GetFloorPlateList.AddtoList oComp
            
            End If
            
        Next i
        
    End If

End Function

Function GetBlockOutList(vFloorPlates As Variant, swView As SldWorks.View) As IArrListObject
    
    Set GetBlockOutList = New IArrListObject

    If Not IsEmpty(vFloorPlates) Then
    
        Dim i As Integer
        For i = LBound(vFloorPlates) To UBound(vFloorPlates)
        
            Dim oFloorPlate As IComp
            Set oFloorPlate = vFloorPlates(i)
            
            Debug.Print oFloorPlate.GetComponent.Name2
            
            Dim vFaces As Variant
            vFaces = swView.GetVisibleEntities2(oFloorPlate.GetComponent, swViewEntityType_e.swViewEntityType_Face)
            
            If Not IsEmpty(vFaces) Then
                
                Dim swFace As SldWorks.Face2
            
                If UBound(vFaces) = 0 Then
                
                   Set swFace = vFaces(0)
                   
                Else
                
                    Set swFace = GetLargestFace(vFaces)
                    
                End If
                
                Dim vLoops As Variant
                vLoops = swFace.GetLoops
                
                Call AddRectangularBlockoutsToList(vLoops, GetBlockOutList, oFloorPlate, swView)

            End If

        Next i
        
    End If
    
End Function

Function GetLargestFace(vFaces As Variant) As SldWorks.Face2

    Dim i As Integer
    Dim Area As Double
    Area = 0
    For i = LBound(vFaces) To UBound(vFaces)
    
        Dim swFace As SldWorks.Face2
        Set swFace = vFaces(i)
        
        If swFace.GetArea > Area Then
        
            Set GetLargestFace = swFace
            Area = swFace.GetArea
            
        End If

    Next i
   
End Function

Sub AddRectangularBlockoutsToList(vLoops As Variant, ArrList As IArrListObject, oComp As IComp, swView As SldWorks.View)

    Dim i As Integer
    For i = LBound(vLoops) To UBound(vLoops)
    
        Dim swLoop As SldWorks.Loop2
        Set swLoop = vLoops(i)
        
        If False = (swLoop.IsOuter) Then
        
            Dim vEdges As Variant
            vEdges = swLoop.GetEdges
            
            If UBound(vEdges) = 3 Then
            
                If Not (IsAnyEdgeHaveNegligibleLength(vEdges)) Then
                
                    If Not (IsContainsCircularEdge(vEdges)) Then
                        
                        Dim oBlockOut As IBlockOut
                        Set oBlockOut = New IBlockOut
    
                        oBlockOut.Initialize swLoop, oComp.GetComponent, swView
    
                        ArrList.AddtoList oBlockOut
                    
                    End If
                    
                End If
            
            End If

        End If
    
    Next i

End Sub

Function IsAnyEdgeHaveNegligibleLength(vEdges As Variant) As Boolean

    IsAnyEdgeHaveNegligibleLength = False
    
    Dim i As Integer
    For i = LBound(vEdges) To UBound(vEdges)
    
        Dim swEdge As SldWorks.Edge
        Set swEdge = vEdges(i)
        
        If GetEdgeLength(swEdge) < 0.1 * 0.0254 Then
            
            IsAnyEdgeHaveNegligibleLength = True
            Exit For
            
        End If

    Next i
    
End Function

Function IsContainsCircularEdge(vEdges As Variant) As Boolean

    IsContainsCircularEdge = False
    
    Dim i As Integer
    For i = LBound(vEdges) To UBound(vEdges)
    
        Dim swEdge As SldWorks.Edge
        Set swEdge = vEdges(i)
        
        Dim swCurve As SldWorks.Curve
        Set swCurve = swEdge.GetCurve
        
        If swCurve.IsCircle Then
        
            IsContainsCircularEdge = True
            Exit For
            
        End If
    
    Next i
    
End Function

Sub AddOrdinateToHorizontalPerimeterBeams(ArrList As IArrListObject, swDrawing As SldWorks.DrawingDoc, _
        swView As SldWorks.View, IsTop As Boolean)
        
    If ArrList.Count > 0 Then
        
        Dim vItems As Variant
        vItems = ArrList.Items
        
        swDrawing.ClearSelection2 True
        swDrawing.SetPickMode
        
        Dim i As Integer
        For i = LBound(vItems) To UBound(vItems)
        
            Dim oBeam As IBlockOut
            Set oBeam = vItems(i)
            
            Dim BeamLeftEdge As SldWorks.Edge
            Set BeamLeftEdge = GetEdgeInViewForBody(oBeam.GetComponent, oBeam, swView, False, False)

            swView.SelectEntity BeamLeftEdge, True

            Dim yPos As Double
        
            If IsTop Then
            
                yPos = oBeam.yMax + 0.01
            
            Else
        
                yPos = oBeam.yMin - 0.01
            
            End If
  
            Dim vComps As Variant

            If IsTop Then
            
                vComps = oBeam.BeforeConnectingPlates
                
            Else
            
                vComps = oBeam.AfterConnectingPlates
                
            End If
            
            
            Call SelectFirstFaceofConnectingPlates(vComps, swView, False)
            
            If i = UBound(vItems) Then
            
                Dim BeamRightEdge As SldWorks.Edge
                Set BeamRightEdge = GetEdgeInViewForBody(oBeam.GetComponent, oBeam, swView, False, True)
            
                swView.SelectEntity BeamRightEdge, True
                
                swDrawing.Extension.AddOrdinateDimension swAddOrdinateDims_e.swHorizontalOrdinate, oBeam.xMax, yPos, 0
                Call AddBracketsAndSuffixToSelectedDimension(swDrawing, "SEE NOTE 3")
                
            End If
        
        
        Next i
    
    End If

End Sub
Private Sub GetViewMaxMinPoints(oComp As IComp, swView As SldWorks.View, ByRef xMin As Double, _
                ByRef xMax As Double, ByRef yMin As Double, ByRef yMax As Double)

    Dim vViewMaxPt As Variant
    vViewMaxPt = GetComponentPointInViewSpace(oComp.GetComponent, oComp.GetMaxPointInModel, swView)
            
    Dim vViewMinPt As Variant
    vViewMinPt = GetComponentPointInViewSpace(oComp.GetComponent, oComp.GetMinPointInModel, swView)
    
    Call StrucutralElevation.GetMaxMinPoint(vViewMinPt(0), vViewMaxPt(0), xMin, xMax)
    Call StrucutralElevation.GetMaxMinPoint(vViewMinPt(1), vViewMaxPt(1), yMin, yMax)
    
End Sub

Function EditTemplate(swDrawing As SldWorks.DrawingDoc, swSheet As SldWorks.Sheet, WeldmentNo As String, SheetName As String)
    
    Dim swSelect As SldWorks.SelectionMgr
    Set swSelect = swDrawing.SelectionManager()
    
    swDrawing.EditTemplate

    Dim SheetFormatName As String
    SheetFormatName = swSheet.GetSheetFormatName
    Dim Boolstatus As Boolean
    
    Dim swNote As INote
    Boolstatus = swDrawing.Extension.SelectByID2("DetailItem1227@" & SheetFormatName, "NOTE", 0.355252206998469, 3.32049059009041E-02, 0, False, 0, Nothing, 0)
    Set swNote = swSelect.GetSelectedObject6(1, -1)
    swNote.SetText (SheetName)
        
    Boolstatus = swDrawing.Extension.SelectByID2("DetailItem1229@" & SheetFormatName, "NOTE", 0.355252206998469, 3.32049059009041E-02, 0, False, 0, Nothing, 0)
    Set swNote = swSelect.GetSelectedObject6(1, -1)
    swNote.SetText (WeldmentNo)
    
    Boolstatus = swDrawing.Extension.SelectByID2("DetailItem1262@" & SheetFormatName, "NOTE", 4.69556851111171E-02, 3.48062939323501E-02, 0, False, 0, Nothing, 0)
    swDrawing.EditDelete
        
    swDrawing.EditSheet
    
End Function

Function AddNoteToView(swDrawing As SldWorks.DrawingDoc, NoteText As String, xPos As Double, yPos As Double) As SldWorks.Note
            
    Set AddNoteToView = swDrawing.InsertNote(NoteText)
            
    If Not AddNoteToView Is Nothing Then

        Dim swAnnotation As SldWorks.Annotation
        Set swAnnotation = AddNoteToView.GetAnnotation()

        If Not swAnnotation Is Nothing Then

            swAnnotation.SetPosition xPos, yPos, 0

        End If

    End If
    
End Function
 
Private Sub AddSeeNote2Circle(swDrawing As SldWorks.DrawingDoc, swView As SldWorks.View, _
        vHorBeams As Variant, vRightBeams As Variant, Optional IsTop As Boolean = True)
    
    Dim oBeam As IBlockOut
    Set oBeam = vHorBeams(UBound(vHorBeams))
    
    Dim RightBeam As IBlockOut
    Set RightBeam = vRightBeams(0)
    
    Dim vShCenterPt(2) As Double
    vShCenterPt(0) = (RightBeam.xMin + RightBeam.xMax) / 2
    vShCenterPt(1) = (oBeam.yMin + oBeam.yMax) / 2
    vShCenterPt(2) = 0

    Dim vShEndPt(2) As Double
    vShEndPt(0) = oBeam.xMax
    vShEndPt(2) = oBeam.zMin
    
    Dim YClearnace As Double
    
    If IsTop Then
        
        vShEndPt(1) = oBeam.yMax
        vShCenterPt(1) = vShCenterPt(1) - 1.75 * 0.0254 * swView.ScaleDecimal
        YClearnace = -0.005
        
    Else
    
        vShEndPt(1) = oBeam.yMin
        vShCenterPt(1) = vShCenterPt(1) + 1.75 * 0.0254 * swView.ScaleDecimal
         YClearnace = 0.0075
    
    End If
    
    Dim vViewCenterPt As Variant
    vViewCenterPt = GetSheetPointInViewSpace(swView, vShCenterPt)
            
    Dim vViewEndPt As Variant
    vViewEndPt = GetSheetPointInViewSpace(swView, vShEndPt)

    Dim radius As Double
    radius = Sqr((vViewCenterPt(0) - vViewEndPt(0)) ^ 2 + (vViewCenterPt(1) - vViewEndPt(1)) ^ 2) + 2 * 0.0254
            
    Dim swSketchSegment As SldWorks.SketchSegment
    Set swSketchSegment = swSketchMgr.CreateCircleByRadius(vViewCenterPt(0), vViewCenterPt(1), vViewCenterPt(2), radius)
    swSketchSegment.ConstructionGeometry = True
    
    Dim Bool As Boolean
    Bool = swDrawing.Extension.SelectByID2("Arc" & swSketchSegment.GetID(1), "SKETCHSEGMENT", vShCenterPt(0) + radius * swView.ScaleDecimal, _
            vShCenterPt(1), 0, False, 0, Nothing, 0)
            
    If Bool Then
    
        Call AddNoteToView(swDrawing, "SEE NOTE 2", vShCenterPt(0) + radius * swView.ScaleDecimal + 0.00625, vShCenterPt(1) + YClearnace)
    
    End If
    
End Sub

Private Sub AddHorizontalAssyOrdinate(vBottomBeams As Variant, vTopBeams As Variant, _
    swDrawing As SldWorks.DrawingDoc, swView As SldWorks.View)

    swDrawing.ClearSelection2 True
    swDrawing.SetPickMode
    
    Dim BottomBeam As IBlockOut
    Set BottomBeam = vBottomBeams(0)
    
    Dim TopBeam As IBlockOut
    Set TopBeam = vTopBeams(0)
        
    Dim BeamBottomEdge As SldWorks.Edge
    Set BeamBottomEdge = GetEdgeInViewForBody(BottomBeam.GetComponent, BottomBeam, swView, True, False)
        
    Dim BeamTopEdge As SldWorks.Edge
    Set BeamTopEdge = GetEdgeInViewForBody(TopBeam.GetComponent, TopBeam, swView, True, True)
    
    swView.SelectEntity BeamBottomEdge, False
    swView.SelectEntity BeamTopEdge, True

    swDrawing.Extension.AddOrdinateDimension swAddOrdinateDims_e.swVerticalOrdinate, BottomBeam.xMin - 0.01, BottomBeam.yMin, 0
    Call AddBracketsAndSuffixToSelectedDimension(swDrawing, "SEE NOTE 3")
    
End Sub

Private Sub SetHiddenEdgesVisibleAndRemoveTangentEdges(swView As SldWorks.View, swDrawing As SldWorks.DrawingDoc)

    Dim Bool As Boolean
    Bool = swDrawing.Extension.SelectByID2(swView.Name, "DRAWINGVIEW", 0, 0, 0, False, 0, Nothing, swSelectOption_e.swSelectOptionDefault)
    
    If Bool Then
    
        swDrawing.ViewDisplayHiddengreyed
    
    End If
    
    swView.SetDisplayTangentEdges2 swDisplayTangentEdges_e.swTangentEdgesHidden
    
End Sub



Function CombineArr(ByVal MainArr As Variant, ArrToAdd As Variant)

    Dim i As Integer
    For i = LBound(ArrToAdd) To UBound(ArrToAdd)
    
        ReDim Preserve MainArr(UBound(MainArr) + 1)
        Set MainArr(UBound(MainArr)) = ArrToAdd(i)
        
    Next i
    
    CombineArr = MainArr
    
End Function

'Private Sub AddVerticalBeamOrdinateDimensions(oBeam As Iblockout, IsAfter As Boolean, vComps As Variant, _
'                swDrawing As SldWorks.ModelDoc2, swView As SldWorks.View, _
'                Optional IsSelectEnd As Boolean = True, Optional Clearance As Double = 0.01)
'
'    If Not IsEmpty(vComps) Then
'
'        Dim BeamLeftEdge As SldWorks.Edge
'        Set BeamLeftEdge = GetEdgeInViewForBody(oBeam.GetComponent, oBeam, swView, False, False)
'
'        swDrawing.ClearSelection2 True
'        swDrawing.SetPickMode
'        swView.SelectEntity BeamLeftEdge, False
'
'        Dim yPos As Double
'
'        If IsAfter Then
'
'            yPos = oBeam.yMax + Clearance
'
'        Else
'
'            yPos = oBeam.yMin - Clearance
'
'        End If
'
'        Call SelectFirstFaceofConnectingPlates(vComps, swView, False)
'
'        If IsSelectEnd Then
'
'            Dim BeamRightEdge As SldWorks.Edge
'            Set BeamRightEdge = GetEdgeInViewForBody(oBeam.GetComponent, oBeam, swView, False, True)
'
'            swView.SelectEntity BeamRightEdge, True
'
'        End If
'
'        swDrawing.Extension.AddOrdinateDimension swAddOrdinateDims_e.swHorizontalOrdinate, oBeam.xMax, yPos, 0
'
'    End If
'
'End Sub
'
'Private Sub AddHorizontalBeamOrdinateDimensions(oBeam As Iblockout, IsAfter As Boolean, vComps As Variant, _
'        swDrawing As SldWorks.ModelDoc2, swView As SldWorks.View, Optional IsSelectEnd As Boolean = True, Optional Clearance As Double = 0.01)
'
'    If Not IsEmpty(vComps) Then
'
'        swDrawing.ClearSelection2 True
'        swDrawing.SetPickMode
'
'        Dim BeamBottomEdge As SldWorks.Edge
'        Set BeamBottomEdge = GetEdgeInViewForBody(oBeam.GetComponent, oBeam, swView, True, False)
'
'        swView.SelectEntity BeamBottomEdge, False
'
'
'        Dim xPos As Double
'
'        If IsAfter Then
'
'            xPos = oBeam.xMax + Clearance
'
'        Else
'            xPos = oBeam.xMin - Clearance
'
'        End If
'
'        Call SelectFirstFaceofConnectingPlates(vComps, swView, True)
'
'        If IsSelectEnd Then
'
'            Dim BeamTopEdge As SldWorks.Edge
'            Set BeamTopEdge = GetEdgeInViewForBody(oBeam.GetComponent, oBeam, swView, True, True)
'
'            swView.SelectEntity BeamTopEdge, True
'
'        End If
'
'        swDrawing.Extension.AddOrdinateDimension swAddOrdinateDims_e.swVerticalOrdinate, xPos, oBeam.yMin, 0
'
'    End If
'
'End Sub

Sub UpdateViewLabelPosition(swView As SldWorks.View, yPos As Double)

    Dim swLabelNote As SldWorks.Note
    Set swLabelNote = swView.GetFirstNote
            
    If Not swLabelNote Is Nothing Then
            
        Dim swLabelAnn As SldWorks.Annotation
        Set swLabelAnn = swLabelNote.GetAnnotation
                
        Dim LabelPos As Variant
        LabelPos = swLabelAnn.GetPosition

        swLabelAnn.SetPosition LabelPos(0), yPos, LabelPos(2)
                
    End If
    
End Sub







Function GetViewInASheetByName(swSheet As SldWorks.Sheet, viewName As String) As SldWorks.View

    Dim vViews As Variant
    vViews = swSheet.GetViews
    
    Dim i As Integer
    For i = LBound(vViews) To UBound(vViews)
    
        Dim swView As SldWorks.View
        Set swView = vViews(i)
        
        If swView.Name = viewName Then
            
            Set GetViewInASheetByName = swView
            Exit For
            
        End If
    
    Next i

End Function



Sub AddEllipseAndCreateDetailView(swDrawing As SldWorks.DrawingDoc, swView As SldWorks.View, ArrList As IArrListObject, _
    ByRef LegendAscii As Long, ByRef IsAsciiMaxReached As Boolean, ByRef ViewDict As Scripting.Dictionary, _
    ByRef BodyDict As Scripting.Dictionary, ByRef IsSubWeldmentExists As Boolean)
    
    Dim scaleRatio As Variant
    scaleRatio = swView.scaleRatio

    If ArrList.Count > 0 Then
        
        Dim vItems As Variant
        vItems = ArrList.Items
        
        Dim Clearance As Double
        Clearance = 7.5 * 0.0254 '* swView.ScaleDecimal
        
        Dim i As Integer
        For i = LBound(vItems) To UBound(vItems)
        
            Dim oBlockOut As IBlockOut
            Set oBlockOut = vItems(i)
                   
            If oBlockOut.IsVertical Or False = oBlockOut.IsPerimeter Then
            
                Dim vMinPoint As Variant
                Dim vMaxPoint As Variant
        
                vMinPoint = BodyExtremePointInViewSpace(oBlockOut, swView, False)
                vMaxPoint = BodyExtremePointInViewSpace(oBlockOut, swView, True)
                        
                Dim Width As Double
                Dim vStartPoint(2) As Double
                Dim vEndPoint(2) As Double
        
                If oBlockOut.IsVertical Then
                            
                    Width = Abs(vMaxPoint(0) - vMinPoint(0)) + Clearance
                    vStartPoint(0) = (vMinPoint(0) + vMaxPoint(0)) / 2
                    vStartPoint(1) = vMinPoint(1)
                    vStartPoint(2) = 0
                            
                    vEndPoint(0) = vStartPoint(0)
                    vEndPoint(1) = vMaxPoint(1)
                    vEndPoint(2) = 0
        
                Else
        
                    Width = Abs(vMaxPoint(1) - vMinPoint(1)) + Clearance
                    vStartPoint(0) = vMinPoint(0)
                    vStartPoint(1) = (vMinPoint(1) + vMaxPoint(1)) / 2
                    vStartPoint(2) = 0
                            
                    vEndPoint(0) = vMaxPoint(0)
                    vEndPoint(1) = vStartPoint(1)
                    vEndPoint(2) = 0
                        
                End If
                
                swView.FocusLocked = True
                        
                Dim swSketchSlot As SldWorks.SketchSlot
                Set swSketchSlot = swSketchMgr.CreateSketchSlot(swSketchSlotCreationType_e.swSketchSlotCreationType_line, _
                                swSketchSlotLengthType_e.swSketchSlotLengthType_FullLength, Width, vStartPoint(0), vStartPoint(1), 0, vEndPoint(0), vEndPoint(1), 0, 0, 0, 0, 1, False)
                
                
                Dim swDetailView As SldWorks.View
                Set swDetailView = swDrawing.CreateDetailViewAt3((oBlockOut.xMin + oBlockOut.xMax) / 2, ((oBlockOut.yMin + oBlockOut.yMax) / 2) - 11 * 0.0254, 0, 2, scaleRatio(0), scaleRatio(1), UCase(Chr(LegendAscii)), 0, False)
                
                If Not swDetailView Is Nothing Then
                    
                    swDetailView.UseSheetScale = 1
                
                    Dim swDetailCircle As SldWorks.DetailCircle
                    Set swDetailCircle = swDetailView.GetDetail
                    
                    swDetailCircle.Layer = "FORMAT"
                    
                    If False = IsSubWeldmentExists Then
                        
                        IsSubWeldmentExists = True
                    
                    End If
                    
                    ViewDict.Add swDetailView.Name, swDetailView 'oBlockOut.GetBody.Name
                    BodyDict.Add swDetailView.Name, oBlockOut
                    
                End If
                
                Call GetValidAscii(LegendAscii, IsAsciiMaxReached)
                
            End If
            
        Next i
        
    End If

End Sub
'Private Function BodyExtremePointInViewSpace(oBlockOut As Iblockout, swView As SldWorks.View, IsMax As Boolean) As Variant
'
'    Dim Point(2) As Double
'    If IsMax Then
'
'        Point(0) = oBlockOut.xMax
'        Point(1) = oBlockOut.yMax
'        Point(2) = oBlockOut.zMax
'
'    Else
'
'        Point(0) = oBlockOut.xMin
'        Point(1) = oBlockOut.yMin
'        Point(2) = oBlockOut.zMin
'
'    End If
'
'    BodyExtremePointInViewSpace = GetSheetPointInViewSpace(swView, Point)
'
'End Function



'Function GetVertexPoint(swView As SldWorks.View, swBeam As Iblockout, IsMax As Boolean, _
'        ParamToCheck As String) As SldWorks.Vertex
'
'    Dim swEdge As SldWorks.Edge
'    Set swEdge = GetEdgeInViewForBody(swBeam.GetComponent, swBeam, swView, True, IsMax)
'
'    Dim swStartVertex As SldWorks.Vertex
'    Set swStartVertex = swEdge.GetStartVertex
'
'    Dim swEndVertex As SldWorks.Vertex
'    Set swEndVertex = swEdge.GetEndVertex
'
'    Dim vStartPoint As Variant
'    vStartPoint = swStartVertex.GetPoint
'
'    Dim vEndPoint As Variant
'    vEndPoint = swEndVertex.GetPoint
'
'    vStartPoint = GetComponentPointInSheetSpace(swBeam.GetComponent, vStartPoint, swView)
'    vEndPoint = GetComponentPointInSheetSpace(swBeam.GetComponent, vEndPoint, swView)
'
'    If Abs(CallByName(swBeam, ParamToCheck, VbGet) - vStartPoint(0)) <= 0.0001 Then
'
'        Set GetVertexPoint = swStartVertex
'
'    ElseIf Abs(CallByName(swBeam, ParamToCheck, VbGet) - vEndPoint(0)) <= 0.0001 Then
'
'        Set GetVertexPoint = swEndVertex
'
'    End If
'
'End Function




Function GetEdgeInViewForBody(swComp As SldWorks.Component2, oBody As IBlockOut, swView As SldWorks.View, _
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
    
    Call GetMaxMinPoint(vPointMin(0), vPointMax(0), xMin, xMax)
    Call GetMaxMinPoint(vPointMin(1), vPointMax(1), yMin, yMax)
    
    Dim Idx As Integer
    Dim ValToMatch As Double
    If IsHorizontal Then
        
        Idx = 1
        If IsMax Then
        
            ValToMatch = yMax
            
        Else
        
             ValToMatch = yMin
             
        End If
        
    Else
    
        Idx = 0
        
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
            
            Dim swEdgeBody As SldWorks.Body2
            Set swEdgeBody = swEdge.GetBody
                
            If swEdgeBody.Name = oBody.GetBody.Name Then

                Dim swCurve As SldWorks.Curve
                Set swCurve = swEdge.GetCurve
                
                If swCurve.IsLine Then
                
                    Dim vStartPoint As Variant
                    vStartPoint = swEdge.GetStartVertex.GetPoint
                    vStartPoint = GetComponentPointInSheetSpace(swComp, vStartPoint, swView)
                    
                    Dim vEndPoint As Variant
                    vEndPoint = swEdge.GetEndVertex.GetPoint
                    vEndPoint = GetComponentPointInSheetSpace(swComp, vEndPoint, swView)

                    If Abs(vStartPoint(Idx) - vEndPoint(Idx)) <= 0.00001 And Abs(vStartPoint(Idx) - ValToMatch) <= 0.00001 Then
                        
                        Dim vCurveParam As Variant
                        vCurveParam = swEdge.GetCurveParams2
    
                        If swCurve.GetLength2(vCurveParam(6), vCurveParam(7)) > TempLength Then
                            
                            TempLength = swCurve.GetLength2(vCurveParam(6), vCurveParam(7))
                            Set GetEdgeInViewForBody = swEdge
                            
                        End If
                        
                    End If
                
                End If
                
            End If
            
        Next i

    End If

End Function

 
 Function InsertBalloonAndGetAnnotations(swDrawing As SldWorks.DrawingDoc, AnnXPos As Double, _
        AnnYPos As Double, Optional Qty As Integer = 1) As SldWorks.Annotation
    
    Dim swBalloonParams As SldWorks.BalloonOptions
    Set swBalloonParams = swDrawing.Extension.CreateBalloonOptions()
    swBalloonParams.Size = swBalloonFit_e.swBF_Tightest
    swBalloonParams.Style = swBalloonStyle_e.swBS_SplitCirc

    
    If Qty > 1 Then
    
        swBalloonParams.ShowQuantity = True
        swBalloonParams.QuantityOverride = True
        swBalloonParams.QuantityOverrideValue = CStr(Qty)
                
    End If
    
    Dim swNote As SldWorks.Note
    Set swNote = swDrawing.Extension.InsertBOMBalloon2(swBalloonParams)
     
    If Not swNote Is Nothing Then
    
        'swNote.PropertyLinkedText = "$PRPWLD:" & Chr(34) & "LEGEND" & Chr(34)
        
        Dim Bool As Boolean
        Bool = swNote.SetBomBalloonText(swDetailingNoteTextContent_e.swDetailingNoteTextCustom, "$PRPWLD:" & Chr(34) & "LEGEND" & Chr(34), _
                    swDetailingNoteTextContent_e.swDetailingNoteTextCustom, "$PRPWLD:" & Chr(34) & "ITEM NO" & Chr(34))

        Set InsertBalloonAndGetAnnotations = swNote.GetAnnotation
        InsertBalloonAndGetAnnotations.SetPosition2 AnnXPos, AnnYPos, 0
        
    End If
    
End Function


Private Sub UpdateTopViewPosition(oFloorComp As IComp, swDrawing As SldWorks.DrawingDoc, swView As SldWorks.View)
    
    Call oFloorComp.CheckForUpdateInMaxMinDimensions(swView)
    
    Dim CenterX As Double
    CenterX = (oFloorComp.xMin + oFloorComp.xMax) / 2
    
    Dim CenterY As Double
    CenterY = (oFloorComp.yMin + oFloorComp.yMax) / 2

    Dim viewPosition As Variant
    viewPosition = swView.Position

    viewPosition(0) = viewPosition(0) + (viewPosition(0) - CenterX)
    viewPosition(1) = viewPosition(1) + (viewPosition(1) - CenterY)

    swView.Position = viewPosition
    
    Call oFloorComp.CheckForUpdateInMaxMinDimensions(swView)

End Sub

Private Function SelectAndAddDimension(swEnt1 As SldWorks.Entity, swEnt2 As SldWorks.Entity, swDrawing As SldWorks.ModelDoc2, _
            xPos As Double, yPos As Double, swView As SldWorks.View, Optional IsDual As Boolean = True) As SldWorks.DisplayDimension

    If Not (swEnt1 Is Nothing) And Not (swEnt2 Is Nothing) Then

        swDrawing.ClearSelection2 True
        
        swView.SelectEntity swEnt1, False
        swView.SelectEntity swEnt2, True

        Set SelectAndAddDimension = swDrawing.AddDimension2(xPos, yPos, 0)

        If Not SelectAndAddDimension Is Nothing Then

            SelectAndAddDimension.CenterText = True

            If IsDual Then

                SelectAndAddDimension.SetDual2 False, False

            End If

        End If

    End If

End Function

Private Function AddStructuralNotes(swDrawing As SldWorks.DrawingDoc) As SldWorks.Note
    
    Dim swSheet As SldWorks.Sheet
    Set swSheet = swDrawing.GetCurrentSheet
    
    swDrawing.ActivateSheet swSheet.GetName

    Dim swStructuralNote As SldWorks.Note
    Dim Note As String
    
    Note = "<FONT size=10PTS style=B>NOTES:" & vbCrLf & _
            "<FONT size=8PTS style=R>1. DIMENSION ORIGIN STARTING AT LOWER LEFT CORNER OF FLOOR BEAM." & vbCrLf & _
            "2. MAKE SURE THE 1/4" & Chr(34) & " LOCATING HOLES AT WALL-A LOWER LEFT CORNER FOR EACH FLOOR TOP PLATES."

    Set swStructuralNote = swDrawing.CreateText2(Note, 1.99241243641486E-02, 6.92464210842187E-02, 0, 0, 0)
    swStructuralNote.SetTextJustification swTextJustification_e.swTextJustificationLeft

End Function
'
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
                "C:\FBD\COMMON\BLOCKS\" & ProjectNo & " EXTERNAL ELEVATION KEY.SLDBLK", True, 1, 0)


End Sub

Function RotateAndScaleView(swDrawing As SldWorks.DrawingDoc, swView As SldWorks.View, _
            ViewWidth As Double, ViewHeight As Double) As SldWorks.View
    

    If ViewHeight > ViewWidth Then
        
        swView.Angle = 1.57079632679
        
        Dim TempVal As Double
        TempVal = ViewHeight
        ViewHeight = ViewWidth
        ViewWidth = TempVal
        
    End If

    Dim xScale As Integer
    Dim yScale As Integer
    xScale = GetScaleValue(ViewWidth / (swView.ScaleDecimal * HorizontalMaxDim))
    yScale = GetScaleValue(ViewHeight / (swView.ScaleDecimal * VerticalMaxDim)) '0.20995
    
    Dim IsScaleSet As Boolean
    IsScaleSet = False
    
    If xScale > 0 And yScale > 0 Then
        
        If yScale > xScale Then
            
            IsScaleSet = swView.Sheet.SetScale(1, yScale, True, True)
           
        Else
            
            IsScaleSet = swView.Sheet.SetScale(1, xScale, True, True)
        
        End If
        
    End If

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



Private Sub UserForm_Initialize()

    Set compDict = New Scripting.Dictionary

End Sub
