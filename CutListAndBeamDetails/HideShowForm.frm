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
    
    Call oFloorComp.GetBodiesList(swTopView)
    
    Dim VerticalWeldBodyList As IArrListObject
    Set VerticalWeldBodyList = oFloorComp.VerticalBodyList

    Dim HorizontalWeldBodyList As IArrListObject
    Set HorizontalWeldBodyList = oFloorComp.HorizontalBodyList
    
' Horizontal Weld Body List sorted with xMin and Vertical with yMin
    
    Dim swBottomBeam As IWeldBody
    Dim swTopBeam As IWeldBody
    Call AddPerimeterBeamProperty(HorizontalWeldBodyList, "yMin", "yMax", swBottomBeam, swTopBeam)
    
    Dim swLeftBeam As IWeldBody
    Dim swRightBeam As IWeldBody
    Call AddPerimeterBeamProperty(VerticalWeldBodyList, "xMin", "xMax", swLeftBeam, swRightBeam)
    
    Dim visibleVerticalMinList As IArrListObject
    Set visibleVerticalMinList = GetVisibleBodiesList(VerticalWeldBodyList.Items, "xMin", "xMax")
    
    Dim visibleHorizontalMinList As IArrListObject
    Set visibleHorizontalMinList = GetVisibleBodiesList(HorizontalWeldBodyList.Items, "yMin", "yMax")
    
    Dim visibleVerticalMaxList As IArrListObject
    Set visibleVerticalMaxList = visibleVerticalMinList.Clone
    
    visibleVerticalMaxList.SortItems "xMax", False
    
    Dim visibleHorizontalMaxList As IArrListObject
    Set visibleHorizontalMaxList = visibleHorizontalMinList.Clone
    
    visibleHorizontalMaxList.SortItems "yMax", False
    
    Dim xMinDict As Scripting.Dictionary
    Dim xMinIndexDict As Scripting.Dictionary
    Set xMinDict = GetPointDataWithIndex(visibleVerticalMinList, "xMin", xMinIndexDict)
    
    Dim xMaxDict As Scripting.Dictionary
    Dim xMaxIndexDict As Scripting.Dictionary
    Set xMaxDict = GetPointDataWithIndex(visibleVerticalMaxList, "xMax", xMaxIndexDict)
    
    Dim yMinDict As Scripting.Dictionary
    Dim yMinIndexDict As Scripting.Dictionary
    Set yMinDict = GetPointDataWithIndex(visibleHorizontalMinList, "yMin", yMinIndexDict)
    
    Dim yMaxDict As Scripting.Dictionary
    Dim yMaxIndexDict As Scripting.Dictionary
    Set yMaxDict = GetPointDataWithIndex(visibleHorizontalMaxList, "yMax", yMaxIndexDict)

    Call FindAndAddBeforeSubWeldments(xMinDict, xMinIndexDict, visibleHorizontalMinList.Clone, "xMin")
    Call FindAndAddAfterSubWeldments(xMaxDict, xMaxIndexDict, visibleHorizontalMinList.Clone, "xMax")

    Call FindAndAddBeforeSubWeldments(yMinDict, yMinIndexDict, visibleVerticalMinList.Clone, "yMin")
    Call FindAndAddAfterSubWeldments(yMaxDict, yMaxIndexDict, visibleVerticalMinList.Clone, "yMax")

    Call AddVerticalCallouts(visibleVerticalMinList, xMinIndexDict, xMinDict, xMaxIndexDict, xMaxDict, swDrawing, swTopView)
    Call AddHorizontalCallouts(visibleHorizontalMinList, yMinIndexDict, yMinDict, yMaxIndexDict, yMaxDict, swDrawing, swTopView)

    Dim LegendAscii As Long
    LegendAscii = 65
    
    Dim IsAsciiMaxReached As Boolean
    IsAsciiMaxReached = False
    
    swApp.SetUserPreferenceToggle swUserPreferenceToggle_e.swSketchInference, False
    
    Dim swTopEdge As SldWorks.Edge
    Dim swBottomEdge As SldWorks.Edge
    
    
    Dim VerticalSubWeldmentList As IArrListObject
    Set VerticalSubWeldmentList = GetSubWeldmentList(visibleVerticalMinList)
    
    Dim HorizontalSubWeldmentList As IArrListObject
    Set HorizontalSubWeldmentList = GetSubWeldmentList(visibleHorizontalMinList)
    
    Dim HorizontalCPlateList As IArrListObject
    Set HorizontalCPlateList = New IArrListObject
    
    Dim VerticalCPlateList As IArrListObject
    Set VerticalCPlateList = New IArrListObject
    
    Call GetHorizontalAndVerticalConnectingPlateList(HorizontalCPlateList, VerticalCPlateList, swTopView)
    
    Call AddConnectingPlatesToSubWeldment(VerticalSubWeldmentList, HorizontalCPlateList)
    Call AddConnectingPlatesToSubWeldment(HorizontalSubWeldmentList, VerticalCPlateList)
    
    Call AddVerticalBeamOrdinateDimensions(swBottomBeam, False, swBottomBeam.AfterConnectingPlates, swDrawing, swTopView)
    Call AddBracketsAndSuffixToSelectedDimension(swDrawing, "SEE NOTE 3")
    
    Call AddVerticalBeamOrdinateDimensions(swTopBeam, True, swTopBeam.BeforeConnectingPlates, swDrawing, swTopView)
    Call AddBracketsAndSuffixToSelectedDimension(swDrawing, "SEE NOTE 3")
    
    Call AddHorizontalAssyOrdinate(swBottomBeam, swTopBeam, swDrawing, swTopView)
    
    Call AddSeeNote2Circle(swDrawing, swTopView, swTopBeam, swRightBeam)
    Call AddSeeNote2Circle(swDrawing, swTopView, swBottomBeam, swRightBeam, False)
    
    swDrawing.ClearSelection2 True
    Call AddNoteToView(swDrawing, "<FONT size=10PTS style=B>TOP VIEW", _
        (swBottomBeam.xMax + swBottomBeam.xMin) / 2, swBottomBeam.yMin - 0.025)
    
    Dim FloorWeldComp As SldWorks.Component2
    Set FloorWeldComp = swFloorWeldment
    
    Call AddViewAndWeldTable(FloorWeldComp, swDrawing, swTopBeam.yMax + 0.015)
    
    Dim SubWeldmentViewDict As Scripting.Dictionary
    Set SubWeldmentViewDict = New Scripting.Dictionary
    
    Dim SubWeldBodyDict As Scripting.Dictionary
    Set SubWeldBodyDict = New Scripting.Dictionary
    
    Dim IsSubWeldmentExists As Boolean
    IsSubWeldmentExists = False
    
    Call AddEllipseAndCreateDetailView(swDrawing, swTopView, VerticalSubWeldmentList, LegendAscii, IsAsciiMaxReached, SubWeldmentViewDict, SubWeldBodyDict, IsSubWeldmentExists)
    Call AddEllipseAndCreateDetailView(swDrawing, swTopView, HorizontalSubWeldmentList, LegendAscii, IsAsciiMaxReached, SubWeldmentViewDict, SubWeldBodyDict, IsSubWeldmentExists)
    
    Call EditTemplate(swDrawing, swDrawing.GetCurrentSheet, WeldmentNo, "CUTLIST AND BEAM DETAILS")
 
    Call AddStructuralNotes(swDrawing, IsSubWeldmentExists)
    Call AddDiagonalDimensionAndNote(swDrawing, swTopView, swBottomBeam, swBottomEdge, swTopBeam, swTopEdge)
    
    Call SetHiddenEdgesVisibleAndRemoveTangentEdges(swTopView, swDrawing)

    Dim swWeldmentSheet As SldWorks.Sheet
    Set swWeldmentSheet = CreateSheetAndMoveDrawingViews(swDrawing, SubWeldmentViewDict, SubWeldBodyDict, WeldmentNo)
    
    Call InsertSketchBlock(swDrawing, swWeldmentSheet, ProjectNo)


    

'    Dim ConsolidatedVerticalBeamList As Scripting.Dictionary
'    Set ConsolidatedVerticalBeamList = GetConsolidatedBeamListOnly(VerticalWeldBodyList, "xMin", "xMax")
'
'    Call FindAndAddSubWeldments(ConsolidatedVerticalBeamList, HorizontalWeldBodyList.Clone, "xMin")
'
'    Dim ConsolidatedHorizontalBeamList As Scripting.Dictionary
'    Set ConsolidatedHorizontalBeamList = GetConsolidatedBeamListOnly(HorizontalWeldBodyList, "yMin", "yMax")
'
'    Call FindAndAddSubWeldments(ConsolidatedHorizontalBeamList, VerticalWeldBodyList.Clone, "yMin")
    

    Set oFloorComp = Nothing
    Set swFloorWeldment = Nothing

    swApp.SetUserPreferenceToggle swUserPreferenceToggle_e.swSketchInference, True
    
    Unload Me

End Sub


Private Sub AddViewAndWeldTable(swComp As SldWorks.Component2, swDrawing As SldWorks.DrawingDoc, ViewMaxLoc As Double)

    Dim swDummyInsView As SldWorks.View
    Set swDummyInsView = swDrawing.CreateDrawViewFromModelView3(swComp.GetModelDoc2().GetPathName(), "*Front", 0.206, 0.296, 0)
        
    If Not swDummyInsView Is Nothing Then
        
        Dim swWeldTableAnn As SldWorks.WeldmentCutListAnnotation
        Set swWeldTableAnn = swDummyInsView.InsertWeldmentTable(False, 0.01590679, SheetBorderTop, _
                    swBOMConfigurationAnchorType_e.swBOMConfigurationAnchor_TopLeft, "", "C:\FBD\COMMON\FBD Templates\CUTLIST TABLE.sldwldtbt")
                    
        If Not swWeldTableAnn Is Nothing Then
            
            Dim swTableAnn As SldWorks.TableAnnotation
            Set swTableAnn = swWeldTableAnn
                
            Dim swAnn As SldWorks.Annotation
            Set swAnn = swTableAnn.GetAnnotation
                
            swAnn.Select3 False, Nothing
            
            'swTableAnn.MoveColumn 0, swTableItemInsertPosition_e.swTableItemInsertPosition_After, 1
                
            swWeldTableAnn.Sort 1, True
            'swTableAnn.MoveColumn 1, swTableItemInsertPosition_e.swTableItemInsertPosition_Before, 0

            Call SplitTableIfNeeded(swTableAnn, ViewMaxLoc)

        End If
        
    End If

End Sub
Private Sub SplitTableIfNeeded(swTableAnn As SldWorks.TableAnnotation, ViewMaxLoc As Double)
    

    Dim TableWidth As Double
    TableWidth = setandGetColumnWidth(swTableAnn)

    Dim rowHeight As Double
    rowHeight = swTableAnn.GetRowHeight(0)
    Debug.Print swTableAnn.Text(1, 2)
    
    Dim ViewTopGap As Double
    ViewTopGap = SheetBorderTop - ViewMaxLoc - 0.01
    
    
    Dim i As Integer
    Dim NoOfRows As Integer
    NoOfRows = Int(ViewTopGap / rowHeight)
        
    Dim MaxNoOfSplits As Integer
    MaxNoOfSplits = Int((0.41595679 - 0.01590679) / TableWidth)
        
    If Int(swTableAnn.RowCount / NoOfRows) < MaxNoOfSplits Then
            
        MaxNoOfSplits = Int(swTableAnn.RowCount / NoOfRows)
            
    Else
            
        NoOfRows = Int(swTableAnn.RowCount / (MaxNoOfSplits + 1)) + 1
            
    End If
        
    If Abs(swTableAnn.RowCount - NoOfRows) > 2 Then
        
        For i = 1 To MaxNoOfSplits
    
            Set swTableAnn = swTableAnn.Split(swTableSplitLocations_e.swTableSplit_AfterRow, i * (NoOfRows - 1))
                    
            If Not swTableAnn Is Nothing Then
                    
                Dim swAnn As SldWorks.Annotation
                Set swAnn = swTableAnn.GetAnnotation()
                        
                swAnn.SetPosition2 0.01590679 + i * (TableWidth + 0.005), SheetBorderTop, 0
                        
            End If
 
        Next i
            
    End If

End Sub

Private Function setandGetColumnWidth(swTable As SldWorks.TableAnnotation) As Double
    
    setandGetColumnWidth = 0
    swTable.SetRowHeight swTableCellRangeIdentifier_e.swTableCellRange_All, 0.004, _
        swTableRowColSizeChangeBehavior_e.swTableRowColChange_TableSizeCanChange
    Const SingleTextWidth = 0.0028
    
    Dim i As Integer
    For i = 0 To swTable.ColumnCount - 1
        
        swTable.setColumnWidth i, SingleTextWidth * Len(swTable.Text(0, i)), _
                swTableRowColSizeChangeBehavior_e.swTableRowColChange_TableSizeCanChange
                
        setandGetColumnWidth = setandGetColumnWidth + swTable.GetColumnWidth(i)
        
    Next i

End Function
Private Function GetColIdx(ColName As String, swTable As SldWorks.TableAnnotation)

    Dim i As Integer
    For i = 0 To swTable.ColumnCount - 1
        
        If swTable.Text(0, i) = ColName Then
        
            GetColIdx = i
            Exit For
            
        End If
    
    Next i
    
End Function
Private Function GetTableWidth(swTable As SldWorks.TableAnnotation) As Double

    GetTableWidth = 0
    
    Dim i As Integer
    For i = 0 To swTable.ColumnCount - 1
        
        Debug.Print swTable.GetColumnWidth(i)
        GetTableWidth = GetTableWidth + swTable.GetColumnWidth(i)
            
    Next i
    
End Function
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
        oBeam As IWeldBody, RightBeam As IWeldBody, Optional IsTop As Boolean = True)
        
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
Private Sub AddBracketsAndSuffixToSelectedDimension(swDrawing As SldWorks.DrawingDoc, Optional suffixNote As String = "")

    Dim swSelectionMgr As SldWorks.SelectionMgr
    Set swSelectionMgr = swDrawing.SelectionManager
    
    If swSelectionMgr.GetSelectedObjectType3(1, -1) = swSelectType_e.swSelDIMENSIONS Then
    
        Dim swDisplayDim As SldWorks.DisplayDimension
        Set swDisplayDim = swSelectionMgr.GetSelectedObject6(1, -1)
        
        swDisplayDim.SetText swDimensionTextParts_e.swDimensionTextPrefix, "("
        swDisplayDim.SetText swDimensionTextParts_e.swDimensionTextSuffix, ")" & vbCrLf & suffixNote
    
    End If

End Sub
Private Sub AddHorizontalAssyOrdinate(BottomBeam As IWeldBody, TopBeam As IWeldBody, _
    swDrawing As SldWorks.DrawingDoc, swView As SldWorks.View)

    swDrawing.ClearSelection2 True
    swDrawing.SetPickMode
        
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

Function GetEdgeInView(oComp As IComp, swView As SldWorks.View, _
    IsHorizontal As Boolean, IsMax As Boolean, Optional CheckAllVisibleEdgesOnly As Boolean = True) As SldWorks.Edge
    
    If InStr(oComp.Name, 1080023) > 0 Then
    
        Debug.Print oComp.Name
    
    End If
    
    Dim xMin As Double
    Dim yMin As Double
    Dim xMax As Double
    Dim yMax As Double
    
    Dim vPointMin(2) As Double
    vPointMin(0) = oComp.xMin
    vPointMin(1) = oComp.yMin
    vPointMin(2) = oComp.zMin
    
    Dim vPointMax(2) As Double
    vPointMax(0) = oComp.xMax
    vPointMax(1) = oComp.yMax
    vPointMax(2) = oComp.zMax
    
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
                vStartPoint = GetComponentPointInSheetSpace(swComp, vStartPoint, swView)
                
                Dim vEndPoint As Variant
                vEndPoint = swEdge.GetEndVertex.GetPoint
                vEndPoint = GetComponentPointInSheetSpace(swComp, vEndPoint, swView)
  
                
                If Abs(vStartPoint(Idx) - vEndPoint(Idx)) <= 0.0001 And Abs(vStartPoint(Idx) - ValToMatch) <= 0.0001 And _
                        Abs(vStartPoint(2) - vEndPoint(2)) <= 0.0001 Then
                    
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

Private Sub AddVerticalBeamOrdinateDimensions(oBeam As IWeldBody, IsAfter As Boolean, vComps As Variant, _
                swDrawing As SldWorks.ModelDoc2, swView As SldWorks.View, Optional IsSelectEnd As Boolean = True, Optional Clearance As Double = 0.01)

    If Not IsEmpty(vComps) Then
    
        swDrawing.ClearSelection2 True
        swDrawing.SetPickMode
        
        Dim BeamLeftEdge As SldWorks.Edge
        Set BeamLeftEdge = GetEdgeInViewForBody(oBeam.GetComponent, oBeam, swView, False, False)

        swView.SelectEntity BeamLeftEdge, False

        Dim yPos As Double
        
        If IsAfter Then
            
            yPos = oBeam.yMax + Clearance
            
        Else
        
            yPos = oBeam.yMin - Clearance
            
        End If

    
        Dim i As Integer
        For i = LBound(vComps) To UBound(vComps)
        
            Dim oComp As IComp
            Set oComp = vComps(i)
            
            Dim swEdge As SldWorks.Edge
            Set swEdge = GetEdgeInView(oComp, swView, False, False, False)
            
            swView.SelectEntity swEdge, True
            
        Next i
        
        If IsSelectEnd Then
        
            Dim BeamRightEdge As SldWorks.Edge
            Set BeamRightEdge = GetEdgeInViewForBody(oBeam.GetComponent, oBeam, swView, False, True)
            
            swView.SelectEntity BeamRightEdge, True
            
        End If
        
        swDrawing.Extension.AddOrdinateDimension swAddOrdinateDims_e.swHorizontalOrdinate, oBeam.xMax, yPos, 0
        
    End If

End Sub

Private Sub AddHorizontalBeamOrdinateDimensions(oBeam As IWeldBody, IsAfter As Boolean, vComps As Variant, _
        swDrawing As SldWorks.ModelDoc2, swView As SldWorks.View, Optional IsSelectEnd As Boolean = True, Optional Clearance As Double = 0.01)
        
    If Not IsEmpty(vComps) Then
    
        swDrawing.ClearSelection2 True
        swDrawing.SetPickMode
        
        Dim BeamBottomEdge As SldWorks.Edge
        Set BeamBottomEdge = GetEdgeInViewForBody(oBeam.GetComponent, oBeam, swView, True, False)

        swView.SelectEntity BeamBottomEdge, False
  
        
        Dim xPos As Double
        
        If IsAfter Then
            
            xPos = oBeam.xMax + Clearance
            
        Else
            xPos = oBeam.xMin - Clearance
            
        End If

        Dim i As Integer
        For i = LBound(vComps) To UBound(vComps)
        
            Dim oComp As IComp
            Set oComp = vComps(i)
            
            Dim swEdge As SldWorks.Edge
            Set swEdge = GetEdgeInView(oComp, swView, True, False, False)
            
            swView.SelectEntity swEdge, True
            
            
        Next i
        
        If IsSelectEnd Then
        
            Dim BeamTopEdge As SldWorks.Edge
            Set BeamTopEdge = GetEdgeInViewForBody(oBeam.GetComponent, oBeam, swView, True, True)
            
            swView.SelectEntity BeamTopEdge, True
            
        End If

        swDrawing.Extension.AddOrdinateDimension swAddOrdinateDims_e.swVerticalOrdinate, xPos, oBeam.yMin, 0
    
    End If

End Sub

Private Sub GetHorizontalAndVerticalConnectingPlateList(ByRef HorizontalList As IArrListObject, ByRef VerticalList As IArrListObject, swView As SldWorks.View)
    
    If compDict.Count > 0 Then
        
        Dim vItems As Variant
        vItems = compDict.Items
        
        Dim i As Integer
        For i = LBound(vItems) To UBound(vItems)
        
            Dim swComp As SldWorks.Component2
            Set swComp = vItems(i)
            
            Dim oComp As IComp
            Set oComp = New IComp
            
            oComp.Initialize swComp, swView
            
            Dim xDiff As Double
            Dim yDiff As Double
            
            xDiff = Abs(oComp.xMax - oComp.xMin)
            yDiff = Abs(oComp.yMax - oComp.yMin)
            
            If xDiff > yDiff Then
            
                HorizontalList.AddtoList oComp
                
                
            Else
            
                VerticalList.AddtoList oComp
                Debug.Print oComp.Name
                
            End If

        Next i
        
        VerticalList.SortItems "yMin", False
        HorizontalList.SortItems "xMin", False
        
    End If
End Sub

Private Sub AddConnectingPlatesToSubWeldment(ArrList As IArrListObject, cPlateList As IArrListObject)

    If ArrList.Count > 0 And cPlateList.Count > 0 Then
    
        Dim vBodies As Variant
        vBodies = ArrList.Items

        Dim i As Integer
        For i = LBound(vBodies) To UBound(vBodies)
        
            Dim oWeldBody As IWeldBody
            Set oWeldBody = vBodies(i)

            Call CheckAndAddConnectingPlate(oWeldBody, cPlateList.Items)
        
        Next i
        
        
    End If

End Sub

Private Sub CheckAndAddConnectingPlate(oWeldBody As IWeldBody, vPlates As Variant)

    Dim i As Integer
    Debug.Print oWeldBody.GetBody.Name
    
    For i = LBound(vPlates) To UBound(vPlates)
    
        Dim oComp As IComp
        Set oComp = vPlates(i)
        
        If oWeldBody.IsVertical Then
        
            If oComp.yMin > oWeldBody.yMin And oComp.yMin < oWeldBody.yMax Then
            
                If (oComp.xMin > oWeldBody.xMin And oComp.xMin < oWeldBody.xMax) Then
                
                    oWeldBody.AddToConnectingPlateList True, oComp
                    Debug.Print oComp.Name
                
                ElseIf (oComp.xMax > oWeldBody.xMin And oComp.xMax < oWeldBody.xMax) Then
                
                    oWeldBody.AddToConnectingPlateList False, oComp
                    Debug.Print oComp.Name

                    
                End If
                
            End If
        
        Else
        
            If oComp.xMin > oWeldBody.xMin And oComp.xMin < oWeldBody.xMax Then
        
                If (oComp.yMin > oWeldBody.yMin And oComp.yMin < oWeldBody.yMax) Then
                    
                    oWeldBody.AddToConnectingPlateList True, oComp
                    Debug.Print oComp.Name

                ElseIf (oComp.yMax > oWeldBody.yMin And oComp.yMax < oWeldBody.yMax) Then
        
                    oWeldBody.AddToConnectingPlateList False, oComp
                    Debug.Print oComp.Name
                    
                    
                End If
                
            End If
        
        End If
    
    
    Next i

End Sub

Private Function CreateSheetAndMoveDrawingViews(swDrawing As SldWorks.DrawingDoc, ViewDict As Scripting.Dictionary, _
    BodyDict As Scripting.Dictionary, WeldmentNo As String) As SldWorks.Sheet
    
    swDrawing.ClearSelection2 True
    If ViewDict.Count > 0 Then
    
        Dim VerticalHeight As Double
        VerticalHeight = 0.2
        
        Dim vKeys As Variant
        vKeys = ViewDict.Keys
        
        Dim SheetScaleVal As Double
        SheetScaleVal = 1
        
        Dim i As Integer
        For i = LBound(vKeys) To UBound(vKeys)
        
            Dim swView As SldWorks.View
            Set swView = ViewDict.Item(vKeys(i))
                
            Dim oBody As IWeldBody
            Set oBody = BodyDict.Item(vKeys(i))
                
            Dim TempScale As Integer
            If oBody.IsVertical Then
                
                TempScale = GetScaleValue((oBody.yMax - oBody.yMin) / (swView.ScaleDecimal * VerticalHeight))
                    
            Else
                
                TempScale = GetScaleValue((oBody.xMax - oBody.xMin) / (swView.ScaleDecimal * HorizontalMaxDim))
                    
            End If
    
                
            If TempScale > SheetScaleVal Then
                
                SheetScaleVal = TempScale
                    
            End If

            swDrawing.Extension.SelectByID2 swView.Name, "DRAWINGVIEW", 0, 0, 0, True, -1, Nothing, 0
        
        Next i

        swDrawing.EditCut
        
        Dim Bool As Boolean
        Bool = swDrawing.NewSheet3("Sheet2", 12, 12, 1, SheetScaleVal, False, "C:\FBD\COMMON\FBD Templates\METAL FAB DRAWING.slddrt", 0.4318, 0.2794, "Default")
        
        Dim swSheet As SldWorks.Sheet
        Set swSheet = swDrawing.Sheet("Sheet2")
        
        swDrawing.ActivateSheet swSheet.GetName
        swDrawing.Extension.SelectByID2 swSheet.GetName, "SHEET", 0, 0, 0, False, 0, Nothing, 0
        
        swApp.RunCommand swCommands_e.swCommands_Paste, "Paste Views"
        
        Call UpdateSubWeldmentViewScale(swDrawing, ViewDict, BodyDict, swSheet)
        
        Dim EndPos As Double
        Call UpdateSubWeldmentVerticalViewPosition(swDrawing, ViewDict, BodyDict, swSheet, EndPos)
        Call UpdateSubWeldmentHorizontalViewPosition(swDrawing, ViewDict, BodyDict, swSheet, EndPos)
        
        Call EditTemplate(swDrawing, swSheet, WeldmentNo, "SUB WELDMENT DETAILS")
        
        Dim swTable As SldWorks.TableAnnotation
        Set swTable = swDrawing.InsertTableAnnotation2(False, 0.32448186, 0.01638209, swBOMConfigurationAnchorType_e.swBOMConfigurationAnchor_TopLeft, _
            "C:\FBD\COMMON\FBD Templates\REV TABLE FOR FABSET.sldtbt", 10, 5)
            
        If Not swTable Is Nothing Then
        
            swTable.BorderLineWeight = swLineWeights_e.swLW_THIN
            swTable.GridLineWeight = swLineWeights_e.swLW_THIN
            
        End If
        
        Set CreateSheetAndMoveDrawingViews = swSheet

    End If
    
End Function

Sub UpdateSubWeldmentVerticalViewPosition(swDrawing As SldWorks.DrawingDoc, ViewDict As Scripting.Dictionary, _
            BodyDict As Scripting.Dictionary, swSheet As SldWorks.Sheet, ByRef LeftPos As Double)
    
    Dim vKeys As Variant
    vKeys = ViewDict.Keys
    
    LeftPos = SheetBorderLeft + 0.01
    
    Dim RightPos As Double

    Dim i As Integer
    For i = LBound(vKeys) To UBound(vKeys)

        Dim swView As SldWorks.View
        Set swView = ViewDict.Item(vKeys(i))
        
        Call SetHiddenEdgesVisibleAndRemoveTangentEdges(swView, swDrawing)

        Dim vOutline As Variant
        vOutline = swView.GetOutline
        
        Dim oBody As IWeldBody
        Set oBody = BodyDict.Item(vKeys(i))
        
        Dim IsSelectEnd As Boolean
        IsSelectEnd = True
        
        If oBody.IsVertical Then
        
            RightPos = LeftPos + (vOutline(2) - vOutline(0)) + 0.02
            If oBody.AfterSubWeldments.Count > 0 And oBody.BeforeSubWeldments.Count > 0 Then
   
                IsSelectEnd = False

            End If
            
            Dim vPosition  As Variant
            vPosition = swView.Position
            
            vPosition(0) = vPosition(0) + (LeftPos - vOutline(0))
            vPosition(1) = vPosition(1) + 0.15847919 - ((vOutline(1) + vOutline(3)) / 2)
                
            LeftPos = RightPos + 0.00625
            
            swView.Position = vPosition

            Call oBody.CheckForUpdateInMaxMinDimensions(swView)
            Call UpdateConnectingPlatePositions(oBody.AfterConnectingPlates, swView)
            Call UpdateConnectingPlatePositions(oBody.BeforeConnectingPlates, swView)

            swDrawing.ClearSelection2 True
            swView.FocusLocked = True
            
            Call UpdateViewLabelPosition(swView, oBody.yMin - (oBody.xMax - oBody.xMin) / 2 - 3.75 * 0.0254 * swView.ScaleDecimal - 0.005)
            
            If oBody.AfterSubWeldments.Count > 0 Then
            
                Call AddHorizontalBeamOrdinateDimensions(oBody, True, oBody.AfterConnectingPlates, swDrawing, swView)
                Call AddBracketsAndSuffixToSelectedDimension(swDrawing)
                
            End If
            
            If oBody.BeforeSubWeldments.Count > 0 Then
            
                Call AddHorizontalBeamOrdinateDimensions(oBody, False, oBody.BeforeConnectingPlates, swDrawing, swView, IsSelectEnd)
                If IsSelectEnd Then
                    Call AddBracketsAndSuffixToSelectedDimension(swDrawing)
                End If
                
            End If

            Call AddCalloutsForSubWeldment(oBody.AfterSubWeldments.Items, swDrawing, swView, True, True, True)
            Call AddCalloutsForSubWeldment(oBody.BeforeSubWeldments.Items, swDrawing, swView, True, True, False)
            Call AddCalloutForSubWeldmentMainBody(oBody, swDrawing, swView)
            
        End If

    Next i
    
End Sub

Sub AddCalloutForSubWeldmentMainBody(oWeldBody As IWeldBody, swDrawing As SldWorks.DrawingDoc, swView As SldWorks.View)
    
    Dim swBodyEdge As SldWorks.Edge
    Dim xPos As Double
    Dim yPos As Double
    Dim AnnXPos As Double
    Dim AnnYPos As Double
            
    If oWeldBody.IsVertical Then
    
        Set swBodyEdge = GetEdgeInViewForBody(oWeldBody.GetComponent, oWeldBody, swView, False, True)
        xPos = oWeldBody.xMax
        yPos = oWeldBody.yMin + 0.01
        AnnXPos = xPos + 0.0075
        AnnYPos = yPos
        
    Else
        
        Set swBodyEdge = GetEdgeInViewForBody(oWeldBody.GetComponent, oWeldBody, swView, True, False)
        xPos = oWeldBody.xMin + 0.01
        yPos = oWeldBody.yMin
        AnnXPos = xPos
        AnnYPos = yPos - 0.0075
        
    End If
    
     Call SelectAndInsertBalloon(swBodyEdge, swDrawing, swView, xPos, yPos, AnnXPos, AnnYPos)
    
End Sub

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

Sub UpdateConnectingPlatePositions(vComps As Variant, swView As SldWorks.View)
    
    If Not IsEmpty(vComps) Then
    
        Dim i As Integer
        For i = LBound(vComps) To UBound(vComps)
        
            Dim oComp As IComp
            Set oComp = vComps(i)
            
            Call oComp.CheckForUpdateInMaxMinDimensions(swView)
        
        Next i
        
    End If
    
End Sub

Sub UpdateSubWeldmentHorizontalViewPosition(swDrawing As SldWorks.DrawingDoc, ViewDict As Scripting.Dictionary, _
            BodyDict As Scripting.Dictionary, swSheet As SldWorks.Sheet, EndPos As Double)
    
    Dim vKeys As Variant
    vKeys = ViewDict.Keys
    
    Dim TopPos As Double
    TopPos = SheetBorderTop - 0.02
    
    Dim BottomPos As Double
    

    Dim i As Integer
    For i = LBound(vKeys) To UBound(vKeys)

        Dim swView As SldWorks.View
        Set swView = ViewDict.Item(vKeys(i))
        
        Dim vOutline As Variant
        vOutline = swView.GetOutline
        
        Dim oBody As IWeldBody
        Set oBody = BodyDict.Item(vKeys(i))
        
        Dim IsSelectEnd As Boolean
        IsSelectEnd = True
        
        If False = oBody.IsVertical Then
            
           If oBody.AfterSubWeldments.Count > 0 And oBody.BeforeSubWeldments.Count > 0 Then
            
                BottomPos = TopPos - (vOutline(3) - vOutline(1)) - 0.025
                IsSelectEnd = False
                
            Else
            
                BottomPos = TopPos - (vOutline(3) - vOutline(1)) - 0.02
                
            End If
            
            Dim vPosition  As Variant
            vPosition = swView.Position
            
            vPosition(1) = vPosition(1) + (TopPos - vOutline(3))
            vPosition(0) = vPosition(0) + ((EndPos - 0.00625 + SheetBorderRight) / 2) - ((vOutline(0) + vOutline(2)) / 2)
                
            TopPos = BottomPos - 0.01
            
            swView.Position = vPosition
            
            Call oBody.CheckForUpdateInMaxMinDimensions(swView)
            Call UpdateConnectingPlatePositions(oBody.AfterConnectingPlates, swView)
            Call UpdateConnectingPlatePositions(oBody.BeforeConnectingPlates, swView)

            
            Call UpdateViewLabelPosition(swView, oBody.yMin - 0.025)
            
            If oBody.AfterSubWeldments.Count > 0 Then
            
                Call AddVerticalBeamOrdinateDimensions(oBody, True, oBody.AfterConnectingPlates, swDrawing, swView, IsSelectEnd)
                If IsSelectEnd Then
                    Call AddBracketsAndSuffixToSelectedDimension(swDrawing)
                End If
                
            End If
            
            If oBody.BeforeSubWeldments.Count > 0 Then
            
                Call AddVerticalBeamOrdinateDimensions(oBody, False, oBody.BeforeConnectingPlates, swDrawing, swView)
                Call AddBracketsAndSuffixToSelectedDimension(swDrawing)
                
            End If
            
            Call AddCalloutsForSubWeldment(oBody.AfterSubWeldments.Items, swDrawing, swView, False, True, False)
            Call AddCalloutsForSubWeldment(oBody.BeforeSubWeldments.Items, swDrawing, swView, False, False, True)
            Call AddCalloutForSubWeldmentMainBody(oBody, swDrawing, swView)

        End If

    Next i


End Sub

Sub AddCalloutsForSubWeldment(vBodies As Variant, swDrawing As SldWorks.DrawingDoc, swView As SldWorks.View, _
    IsHorizontal As Boolean, IsUp As Boolean, IsRight As Boolean)
    
    If Not IsEmpty(vBodies) Then
    
        Dim PickPos As Double
        PickPos = (3.5 * 0.0254 * swView.ScaleDecimal) / 2
    
        Dim i As Integer
        For i = LBound(vBodies) To UBound(vBodies)
        
            Dim oWeldBody As IWeldBody
            Set oWeldBody = vBodies(i)
            
            Call oWeldBody.CheckForUpdateInMaxMinDimensions(swView)
    
            Dim swBodyEdge As SldWorks.Edge
            Dim xPos As Double
            Dim yPos As Double
            Dim AnnXPos As Double
            Dim AnnYPos As Double
            
            If IsHorizontal Then
            
                Set swBodyEdge = GetEdgeInViewForBody(oWeldBody.GetComponent, oWeldBody, swView, IsHorizontal, IsUp)
                
                If IsRight Then
                
                    xPos = oWeldBody.xMin + PickPos
                    AnnXPos = xPos + 0.005
                    
                Else
                    
                    xPos = oWeldBody.xMax - PickPos
                    AnnXPos = xPos - 0.0075
                    
                End If
                    
                If IsUp Then
                
                    yPos = oWeldBody.yMax
                    AnnYPos = yPos + 0.00875
                
                Else
                        
                    yPos = oWeldBody.yMin
                    AnnYPos = yPos - 0.005
                
                End If
            
            Else
                
                Set swBodyEdge = GetEdgeInViewForBody(oWeldBody.GetComponent, oWeldBody, swView, IsHorizontal, IsRight)
                
                If IsRight Then
                
                    xPos = oWeldBody.xMax
                    AnnXPos = xPos + 0.005
                    
                Else
                    
                    xPos = oWeldBody.xMin
                    AnnXPos = xPos - 0.00875
                    
                End If
                    
                If IsUp Then
                
                    yPos = oWeldBody.yMin + PickPos
                    AnnYPos = yPos + 0.00875
                
                Else
                        
                    yPos = oWeldBody.yMax - PickPos
                    AnnYPos = yPos - 0.005
                
                End If
            
            
            End If
            
            Call SelectAndInsertBalloon(swBodyEdge, swDrawing, swView, xPos, yPos, AnnXPos, AnnYPos)
            
        Next i
        
    End If

End Sub

Sub SelectAndInsertBalloon(swBodyEdge As SldWorks.Edge, swDrawing As SldWorks.DrawingDoc, swView As SldWorks.View, _
    xPos As Double, yPos As Double, AnnXPos As Double, AnnYPos As Double)

    Dim IsSelected As Boolean
    IsSelected = SelectEdgeWithSelectData(swBodyEdge, swView, swDrawing, xPos, yPos)
        
    If IsSelected Then
                    
        Call InsertBalloonAndGetAnnotations(swDrawing, AnnXPos, AnnYPos)
                        
    End If
    
End Sub


Sub UpdateSubWeldmentViewScale(swDrawing As SldWorks.DrawingDoc, ViewDict As Scripting.Dictionary, _
            BodyDict As Scripting.Dictionary, swSheet As SldWorks.Sheet)
    
    Dim vKeys As Variant
    vKeys = ViewDict.Keys

    Dim i As Integer
    For i = LBound(vKeys) To UBound(vKeys)
    
        Dim swView As SldWorks.View
        Set swView = GetViewInASheetByName(swSheet, CStr(vKeys(i)))

        swView.UseSheetScale = 1
        
        Set ViewDict.Item(vKeys(i)) = swView

    Next i
    
    Dim Bool As Boolean
    Bool = swDrawing.Extension.Rebuild(swRebuildOptions_e.swCurrentSheetDisp)

End Sub

Function GetViewInASheetByName(swSheet As SldWorks.Sheet, ViewName As String) As SldWorks.View

    Dim vViews As Variant
    vViews = swSheet.GetViews
    
    Dim i As Integer
    For i = LBound(vViews) To UBound(vViews)
    
        Dim swView As SldWorks.View
        Set swView = vViews(i)
        
        If swView.Name = ViewName Then
            
            Set GetViewInASheetByName = swView
            Exit For
            
        End If
    
    Next i

End Function



Function GetSubWeldmentList(ArrList As IArrListObject) As IArrListObject

    Set GetSubWeldmentList = New IArrListObject

    If ArrList.Count > 0 Then
        
        Dim vItems As Variant
        vItems = ArrList.Items

        Dim i As Integer
        For i = LBound(vItems) To UBound(vItems)
        
            Dim oWeldBody As IWeldBody
            Set oWeldBody = vItems(i)

            If oWeldBody.IsSubWeldment Then
                    
                GetSubWeldmentList.AddtoList oWeldBody

            End If

        Next i
        
    End If

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
        
            Dim oWeldBody As IWeldBody
            Set oWeldBody = vItems(i)
                   
            If oWeldBody.IsVertical Or False = oWeldBody.IsPerimeter Then
            
                Dim vMinPoint As Variant
                Dim vMaxPoint As Variant
        
                vMinPoint = BodyExtremePointInViewSpace(oWeldBody, swView, False)
                vMaxPoint = BodyExtremePointInViewSpace(oWeldBody, swView, True)
                        
                Dim Width As Double
                Dim vStartPoint(2) As Double
                Dim vEndPoint(2) As Double
        
                If oWeldBody.IsVertical Then
                            
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
                Set swDetailView = swDrawing.CreateDetailViewAt3((oWeldBody.xMin + oWeldBody.xMax) / 2, ((oWeldBody.yMin + oWeldBody.yMax) / 2) - 11 * 0.0254, 0, 2, scaleRatio(0), scaleRatio(1), UCase(Chr(LegendAscii)), 0, False)
                
                If Not swDetailView Is Nothing Then
                    
                    swDetailView.UseSheetScale = 1
                
                    Dim swDetailCircle As SldWorks.DetailCircle
                    Set swDetailCircle = swDetailView.GetDetail
                    
                    swDetailCircle.Layer = "FORMAT"
                    
                    If False = IsSubWeldmentExists Then
                        
                        IsSubWeldmentExists = True
                    
                    End If
                    
                    ViewDict.Add swDetailView.Name, swDetailView 'oWeldBody.GetBody.Name
                    BodyDict.Add swDetailView.Name, oWeldBody
                    
                End If
                
                Call GetValidAscii(LegendAscii, IsAsciiMaxReached)
                
            End If
            
        Next i
        
    End If

End Sub
Private Function BodyExtremePointInViewSpace(oWeldBody As IWeldBody, swView As SldWorks.View, IsMax As Boolean) As Variant

    Dim Point(2) As Double
    If IsMax Then
        
        Point(0) = oWeldBody.xMax
        Point(1) = oWeldBody.yMax
        Point(2) = oWeldBody.zMax
    
    Else
    
        Point(0) = oWeldBody.xMin
        Point(1) = oWeldBody.yMin
        Point(2) = oWeldBody.zMin
        
    End If
    
    BodyExtremePointInViewSpace = GetSheetPointInViewSpace(swView, Point)

End Function

Sub AddDiagonalDimensionAndNote(swDrawing As SldWorks.DrawingDoc, swView As SldWorks.View, swBottomBeam As IWeldBody, ByRef swBottomEdge As SldWorks.Edge, _
    swTopBeam As IWeldBody, ByRef swTopEdge As SldWorks.Edge)

    Dim MinVertex As SldWorks.Vertex
    Set MinVertex = GetVertexPoint(swView, swBottomBeam, False, "xMin", swBottomEdge)

    Dim MaxVertex As SldWorks.Vertex
    Set MaxVertex = GetVertexPoint(swView, swTopBeam, True, "xMax", swTopEdge)
    
    Dim swDisplayDim As SldWorks.DisplayDimension
    Set swDisplayDim = SelectAndAddDimension(MinVertex, MaxVertex, swDrawing, swTopBeam.xMax + 0.005, swTopBeam.yMax, swView, True)
    
    If Not swDisplayDim Is Nothing Then
        
        Dim Note As String
        Note = "<FONT size=8PTS style=R>DIAGONAL DIMENSION:" & Chr(34) & swDisplayDim.GetDimension2(0).Name & "@" & swView.Name & Chr(34)
        
        Dim swNote As SldWorks.Note
        Set swNote = swDrawing.CreateText2(Note, 1.99241243641486E-02, 7.52464210842187E-02, 0, 0, 0)
        
    
        
    End If
    
End Sub

Function GetVertexPoint(swView As SldWorks.View, swBeam As IWeldBody, IsMax As Boolean, ParamToCheck As String, _
            ByRef swEdge As SldWorks.Edge) As SldWorks.Vertex

    Set swEdge = GetEdgeInViewForBody(swBeam.GetComponent, swBeam, swView, True, IsMax)
    
    Dim swStartVertex As SldWorks.Vertex
    Set swStartVertex = swEdge.GetStartVertex
    
    Dim swEndVertex As SldWorks.Vertex
    Set swEndVertex = swEdge.GetEndVertex
    
    Dim vStartPoint As Variant
    vStartPoint = swStartVertex.GetPoint
    
    Dim vEndPoint As Variant
    vEndPoint = swEndVertex.GetPoint
    
    vStartPoint = GetComponentPointInSheetSpace(swBeam.GetComponent, vStartPoint, swView)
    vEndPoint = GetComponentPointInSheetSpace(swBeam.GetComponent, vEndPoint, swView)
    
    If Abs(CallByName(swBeam, ParamToCheck, VbGet) - vStartPoint(0)) <= 0.0001 Then
    
        Set GetVertexPoint = swStartVertex

    ElseIf Abs(CallByName(swBeam, ParamToCheck, VbGet) - vEndPoint(0)) <= 0.0001 Then
    
        Set GetVertexPoint = swEndVertex

    End If

End Function

Function GetValidAscii(LegendAscii As Long, ByRef IsAsciiMaxReached As Boolean) As Long

    Dim IsNotValid As Boolean
    IsNotValid = True
    
    Do While IsNotValid
            
        LegendAscii = LegendAscii + 1
        If Not (LegendAscii = 73 Or LegendAscii = 79 Or LegendAscii = 81 Or LegendAscii = 83 Or LegendAscii = 88 Or LegendAscii = 90) Then
                
            IsNotValid = False
            
            If LegendAscii > 90 Then
            
                IsAsciiMaxReached = True
                LegendAscii = 65
                
            End If
            
        End If
            
    Loop

    GetValidAscii = LegendAscii
    
End Function


Function GetEdgeInViewForBody(swComp As SldWorks.Component2, oBody As IWeldBody, swView As SldWorks.View, _
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

Private Sub AddVerticalCallouts(ArrList As IArrListObject, MinIndexDict As Scripting.Dictionary, MinDict As Scripting.Dictionary, _
            MaxIndexDict As Scripting.Dictionary, MaxDict As Scripting.Dictionary, _
            swDrawing As SldWorks.ModelDoc2, swView As SldWorks.View)

    Dim i As Integer

    Dim vItems As Variant
    vItems = ArrList.Items
    
    For i = LBound(vItems) To UBound(vItems)
    
        Dim oWeldBody As IWeldBody
        Set oWeldBody = vItems(i)
        
        Dim swBodyEdge As SldWorks.Edge
        
        Dim xPos As Double
        Dim yPos As Double

        Dim AnnXPos As Double
        Dim AnnYPos As Double

        If oWeldBody.IsPerimeter Then
        
            If i = LBound(vItems) Then
            
                Call GetCalloutBeforeThisBody(oWeldBody, swView, swBodyEdge, xPos, AnnXPos)
                Call GetCalloutUporDown(oWeldBody, oWeldBody.AfterSubWeldments, yPos, AnnYPos, False)

            Else
                
                Call GetCalloutAfterThisBody(oWeldBody, swView, swBodyEdge, xPos, AnnXPos)
                Call GetCalloutUporDown(oWeldBody, oWeldBody.BeforeSubWeldments, yPos, AnnYPos, True)
                
            End If
            
        Else

            Dim NextGap As Double
            NextGap = GetNextGap(oWeldBody, MinDict, MinIndexDict)
        
            If oWeldBody.AfterBody.IsPerimeter Then
                
                Call GetCalloutAfterThisBody(oWeldBody, swView, swBodyEdge, xPos, AnnXPos)
                yPos = oWeldBody.yMin + 0.95 * (oWeldBody.yMax - oWeldBody.yMin)
                AnnYPos = oWeldBody.AfterBody.yMax + 0.0075
            
            ElseIf oWeldBody.BeforeBody.IsPerimeter Then
            
                Call GetCalloutAfterThisBody(oWeldBody, swView, swBodyEdge, xPos, AnnXPos)
                yPos = oWeldBody.yMin + 0.05 * (oWeldBody.yMax - oWeldBody.yMin)
                AnnYPos = oWeldBody.BeforeBody.yMin - 0.0025
    
            Else

        
                If NextGap > BalloonWidth Then
            
                    Call GetCalloutAfterThisBody(oWeldBody, swView, swBodyEdge, xPos, AnnXPos)
                    Call GetCalloutUporDown(oWeldBody, oWeldBody.AfterSubWeldments, yPos, AnnYPos, True)
                        
                Else
                        
                    Call GetCalloutBeforeThisBody(oWeldBody, swView, swBodyEdge, xPos, AnnXPos)
                    Call GetCalloutUporDown(oWeldBody, oWeldBody.BeforeSubWeldments, yPos, AnnYPos, False)
                        
                End If
                
            End If

        End If
        
        Call SelectAndInsertBalloon(swBodyEdge, swDrawing, swView, xPos, yPos, AnnXPos, AnnYPos)

    Next i
    
End Sub

Private Sub AddHorizontalCallouts(ArrList As IArrListObject, MinIndexDict As Scripting.Dictionary, MinDict As Scripting.Dictionary, _
            MaxIndexDict As Scripting.Dictionary, MaxDict As Scripting.Dictionary, _
            swDrawing As SldWorks.ModelDoc2, swView As SldWorks.View)

    Dim i As Integer

    Dim vItems As Variant
    vItems = ArrList.Items
    
    For i = LBound(vItems) To UBound(vItems)
    
        Dim oWeldBody As IWeldBody
        Set oWeldBody = vItems(i)
        
        Dim swBodyEdge As SldWorks.Edge
        
        Dim xPos As Double
        Dim yPos As Double

        Dim AnnXPos As Double
        Dim AnnYPos As Double

        If oWeldBody.IsPerimeter Then
        
            If i = LBound(vItems) Then
            
                Call GetCalloutBeforeThisBody(oWeldBody, swView, swBodyEdge, yPos, AnnYPos, 0.005, True)
                Call GetCalloutLeftOrRight(oWeldBody, oWeldBody.AfterSubWeldments, xPos, AnnXPos, False)

            Else
                
                Call GetCalloutAfterThisBody(oWeldBody, swView, swBodyEdge, yPos, AnnYPos, 0.01, True)
                Call GetCalloutLeftOrRight(oWeldBody, oWeldBody.BeforeSubWeldments, xPos, AnnXPos, True)
                
            End If
            
        Else

            Dim NextGap As Double
            NextGap = GetNextGap(oWeldBody, MinDict, MinIndexDict, "yMin", "yMax", "xMin", "xMax")
    
            If NextGap > BalloonWidth Then

                If oWeldBody.AfterBody.IsPerimeter Then

                    Call GetCalloutAfterThisBody(oWeldBody, swView, swBodyEdge, yPos, AnnYPos, 0.01, True, NextGap)
                    xPos = oWeldBody.xMax - 0.00375
                    AnnXPos = xPos - 0.00125
            
                ElseIf oWeldBody.BeforeBody.IsPerimeter Then
                
                    Call GetCalloutAfterThisBody(oWeldBody, swView, swBodyEdge, yPos, AnnYPos, 0.01, True, NextGap)
                    xPos = oWeldBody.xMin + 0.00375
                    AnnXPos = xPos
    
                Else
            
                    Call GetCalloutAfterThisBody(oWeldBody, swView, swBodyEdge, yPos, AnnYPos, 0.01, True, NextGap)
                    Call GetCalloutLeftOrRight(oWeldBody, oWeldBody.AfterSubWeldments, xPos, AnnXPos, True, NextGap)
                        
                End If

                
            Else
                        
                Call GetCalloutBeforeThisBody(oWeldBody, swView, swBodyEdge, yPos, AnnYPos, 0.005, True)
                Call GetCalloutLeftOrRight(oWeldBody, oWeldBody.BeforeSubWeldments, xPos, AnnXPos, False)
                        
            End If
            

        End If
        
        Call SelectAndInsertBalloon(swBodyEdge, swDrawing, swView, xPos, yPos, AnnXPos, AnnYPos)

    Next i
    
End Sub


Function GetAnnPos(yPos As Double, IsUp As Boolean, Clearance As Double)
    
    If IsUp Then

        GetAnnPos = yPos + Clearance
        
    Else
    
        GetAnnPos = yPos - Clearance
    
    End If
    
End Function

Function GetPos(MinVal As Double, Gap As Double, IsRight As Boolean, Optional IsRightPercent As Double = 0.95, Optional IsLeftPercent As Double = 0.6)
    
    If IsRight Then
    
        GetPos = MinVal + Gap * IsRightPercent
        
    Else
    
        GetPos = MinVal + Gap * IsLeftPercent
        
    End If
    
End Function

Sub GetCalloutLeftOrRight(oWeldBody As IWeldBody, WeldList As IArrListObject, ByRef Pos As Double, ByRef AnnPos As Double, _
             IsAfter As Boolean, Optional NextGap As Double = 0.1062)
    
    Dim Min As Double
    Dim Max As Double
    
    Const MinDistForTwoBalloons As Double = 0.018
    Call FindMinAndMaxInGap(WeldList, oWeldBody, "xMin", "xMax", Min, Max)

    
    Dim Gap As Double
    Gap = Max - Min
    
    If oWeldBody.IsPerimeter Then
        
        Pos = Min + Gap * 0.65
        AnnPos = Pos
    
    Else

'        If Gap > MinDistForTwoBalloons Then
        If IsAfter Then
        
            Pos = Max - 0.00375
            AnnPos = Pos - 0.0075
            
        Else
        
            Pos = Min + 0.00375
            AnnPos = Pos + 0.0075
        
        
        End If
        
        If NextGap >= 0.01062 Then

             AnnPos = Pos - 0.001375
        
        End If
'
'            Pos = GetPos(Min, Gap, IsAfter)
'            AnnPos = GetAnnPos(Pos, False, 0.005)

'        Else
'
'            Pos = GetPos(Min, Gap, IsAfter, 0.05, 0.95)
'            AnnPos = oWeldBody.AfterBody.xMax + 0.0025
'
'        End If

    End If
'


End Sub

Sub GetCalloutUporDown(oWeldBody As IWeldBody, WeldList As IArrListObject, ByRef Pos As Double, ByRef AnnPos As Double, _
             IsAfter As Boolean, Optional Clearance As Double = 0.01)
    
    Dim Min As Double
    Dim Max As Double
    
    Const MinDistForTwoBalloons As Double = 0.018
    Call FindMinAndMaxInGap(WeldList, oWeldBody, "yMin", "yMax", Min, Max)

    
    Dim Gap As Double
    Gap = Max - Min
    
    If oWeldBody.IsPerimeter Then
        
        Pos = Min + Gap * 0.5
        AnnPos = GetAnnPos(Pos, True, 0.01)
    
    Else

        If Gap > MinDistForTwoBalloons Then

            Pos = GetPos(Min, Gap, IsAfter)
            AnnPos = GetAnnPos(Pos, False, 0.005)

        Else

            Pos = GetPos(Min, Gap, IsAfter)
            AnnPos = oWeldBody.AfterBody.yMax + 0.01

        End If

    End If

End Sub
 
Sub GetCalloutBeforeThisBody(oWeldBody As IWeldBody, swView As SldWorks.View, ByRef swBodyEdge As SldWorks.Edge, _
            ByRef Pos As Double, ByRef AnnPos As Double, Optional Clearance As Double = 0.005125, Optional IsHorizontal As Boolean = False)
 
    Set swBodyEdge = GetEdgeInViewForBody(oWeldBody.GetComponent, oWeldBody, swView, IsHorizontal, False)
    
    If IsHorizontal Then
    
        Pos = oWeldBody.yMin
        
    Else
    
        Pos = oWeldBody.xMin

    End If
    
    AnnPos = Pos - Clearance
 
End Sub

Sub GetCalloutAfterThisBody(oWeldBody As IWeldBody, swView As SldWorks.View, ByRef swBodyEdge As SldWorks.Edge, _
    ByRef Pos As Double, ByRef AnnPos As Double, Optional Clearance As Double = 0.00375, Optional IsHorizontal As Boolean = False, Optional NextGap As Double = 0.0106)
                     
    Set swBodyEdge = GetEdgeInViewForBody(oWeldBody.GetComponent, oWeldBody, swView, IsHorizontal, True)
    
    If IsHorizontal Then
    
        Pos = oWeldBody.yMax
        AnnPos = Pos + Clearance
        If NextGap < 0.0106 Then
            
            AnnPos = Pos + 0.9 * NextGap
            
        End If
        
    Else
    
        Pos = oWeldBody.xMax
        AnnPos = Pos + Clearance
         
    End If
    
   

End Sub
 
Function GetNextGap(oWeldBody As IWeldBody, MinDict As Scripting.Dictionary, MinIndexDict As Scripting.Dictionary, Optional MinParam As String = "xMin", _
    Optional MaxParam As String = "xMax", Optional CheckMinParam As String = "yMin", Optional CheckMaxParam As String = "yMax")

    Dim Idx As Integer
    Idx = GetIndex(MinIndexDict, oWeldBody, MinParam)
    
    Dim IsFound As Boolean
    IsFound = False
    
    While Not IsFound
    
        Idx = Idx + 1
        Dim NextArrList As IArrListObject
        Set NextArrList = MinDict.Item(MinDict.Keys(Idx))
        
        Dim vItems As Variant
        vItems = NextArrList.Items
        
        Dim i As Integer
        For i = LBound(vItems) To UBound(vItems)
        
            Dim NextWeldBody As IWeldBody
            Set NextWeldBody = vItems(i)
            
            If (((CallByName(NextWeldBody, CheckMinParam, VbGet) < CallByName(oWeldBody, CheckMinParam, VbGet) Or _
                Abs(CallByName(NextWeldBody, CheckMinParam, VbGet) - CallByName(oWeldBody, CheckMinParam, VbGet)) <= 0.0001) And _
                CallByName(oWeldBody, CheckMinParam, VbGet) < CallByName(NextWeldBody, CheckMaxParam, VbGet)) Or _
                ((CallByName(NextWeldBody, CheckMinParam, VbGet) > CallByName(oWeldBody, CheckMinParam, VbGet) Or _
                Abs(CallByName(NextWeldBody, CheckMinParam, VbGet) - CallByName(oWeldBody, CheckMinParam, VbGet)) <= 0.0001) And _
                (CallByName(NextWeldBody, CheckMinParam, VbGet) < CallByName(oWeldBody, CheckMaxParam, VbGet)))) And _
                 (((NextWeldBody.zMin < oWeldBody.zMin Or Abs(NextWeldBody.zMin - oWeldBody.zMin) <= 0.0001) And NextWeldBody.zMax > oWeldBody.zMin) Or _
                 ((NextWeldBody.zMin > oWeldBody.zMin Or Abs(NextWeldBody.zMin - oWeldBody.zMin) <= 0.0001) And (NextWeldBody.zMin < oWeldBody.zMax))) Then
                
                IsFound = True
                GetNextGap = CallByName(NextWeldBody, MinParam, VbGet) - CallByName(oWeldBody, MaxParam, VbGet)
                Exit For
                
            End If
        
        
        Next i
        
    Wend

End Function

 Sub FindMinAndMaxInGap(ArrList As IArrListObject, oWeldBody As IWeldBody, MinParam As String, MaxParam As String, _
    ByRef MinVal As Double, ByRef MaxVal As Double)

    If ArrList.Count = 0 Then
    
        MinVal = CallByName(oWeldBody, MinParam, VbGet)
        MaxVal = CallByName(oWeldBody, MaxParam, VbGet)

    Else

        Dim vItems As Variant
        vItems = ArrList.Items
        
        Dim i As Integer
        
        Dim Gap As Double
        Gap = Abs(CallByName(vItems(0), MinParam, VbGet) - CallByName(oWeldBody, MinParam, VbGet))
        
        MinVal = CallByName(oWeldBody, MinParam, VbGet)
        MaxVal = CallByName(vItems(0), MinParam, VbGet)
       
        Dim TempGap As Double
        Dim TempMinVal As Double
        Dim TempMaxVal As Double

        For i = LBound(vItems) To UBound(vItems)
        
            Dim oSubWeldBody As IWeldBody
            Set oSubWeldBody = vItems(i)

            If i = UBound(vItems) Then
                
                TempMinVal = CallByName(oSubWeldBody, MaxParam, VbGet)
                TempMaxVal = CallByName(oWeldBody, MaxParam, VbGet)
                
            Else
            
                Dim NextWeldBody As IWeldBody
                Set NextWeldBody = vItems(i + 1)
            
                
                TempMinVal = CallByName(oSubWeldBody, MaxParam, VbGet)
                TempMaxVal = CallByName(NextWeldBody, MinParam, VbGet)
                
            End If
            
            TempGap = TempMaxVal - TempMinVal

            If TempGap > Gap Then
                
                Gap = TempGap
                MinVal = TempMinVal
                MaxVal = TempMaxVal
                
            End If
            
        Next i
        
    End If
 
 End Sub
 
 Function SelectEdgeWithSelectData(swEdge As SldWorks.Edge, swView As SldWorks.View, swDrawing As SldWorks.DrawingDoc, _
                SelXPos As Double, SelYPos As Double) As Boolean

    Dim swSelectMgr As SldWorks.SelectionMgr
    Set swSelectMgr = swDrawing.SelectionManager
    
    Dim swSelectData As SldWorks.SelectData
    Set swSelectData = swSelectMgr.CreateSelectData


    swSelectData.View = swView
    swSelectData.X = SelXPos '(vStartPoint(0) + vEndPoint(0)) / 2
    swSelectData.Y = SelYPos 'vStartPoint(1)
    
    Dim swEntity As SldWorks.Entity
    Set swEntity = swEdge

    SelectEdgeWithSelectData = swEntity.Select4(False, swSelectData)
    
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


Function GetIndex(Dict As Scripting.Dictionary, oWeldBody As IWeldBody, Param As String) As Integer
    
    Dim keyVal As Double
    keyVal = CallByName(oWeldBody, Param, VbGet)
    
    If Dict.Exists(keyVal) Then
    
        GetIndex = Dict.Item(keyVal)
        
    Else
    
        Dim vKeys As Variant
        vKeys = Dict.Keys
        
        Dim i As Integer
        
        For i = LBound(vKeys) To UBound(vKeys)

            If Abs(vKeys(i) - keyVal) <= 0.0001 Then
            
                GetIndex = i
                Exit For
                
            End If
            
        Next i
        
    End If

End Function

Function GetPointDataWithIndex(ArrList As IArrListObject, Parameter As String, ByRef IndexDict As Scripting.Dictionary) As Scripting.Dictionary

    Set GetPointDataWithIndex = New Scripting.Dictionary
    Set IndexDict = New Scripting.Dictionary
    
    Dim i As Integer
    Dim vItems As Variant
    vItems = ArrList.Items
    
    For i = LBound(vItems) To UBound(vItems)
    
        Dim oWeldBody As IWeldBody
        Set oWeldBody = vItems(i)
        
        Dim keyVal As Double
        keyVal = CallByName(oWeldBody, Parameter, VbGet)
        
        If GetPointDataWithIndex.Exists(keyVal) Then
            
             GetPointDataWithIndex.Item(keyVal).AddtoList oWeldBody
        
        Else
            
            Dim NewArrList As IArrListObject
            Set NewArrList = New IArrListObject
            
            NewArrList.AddtoList oWeldBody
            
            If GetPointDataWithIndex.Count = 0 Then

                GetPointDataWithIndex.Add keyVal, NewArrList
                IndexDict.Add keyVal, IndexDict.Count
                
            Else
            
                Dim PrevKey As Double
                PrevKey = GetPointDataWithIndex.Keys(GetPointDataWithIndex.Count - 1)
                
                If Abs(PrevKey - keyVal) <= 0.0001 Then
                    
                    GetPointDataWithIndex.Item(PrevKey).AddtoList oWeldBody
                    
                Else
                
                    GetPointDataWithIndex.Add keyVal, NewArrList
                    IndexDict.Add keyVal, IndexDict.Count
                
                End If

            End If
            
        End If

    Next i
    
End Function

Function GetVisibleBodiesList(vItems As Variant, ParamMin As String, ParamMax As String) As IArrListObject

    Set GetVisibleBodiesList = New IArrListObject
    
    Dim i As Integer
    For i = LBound(vItems) To UBound(vItems)
    
        Dim oWeldBody As IWeldBody
        Set oWeldBody = vItems(i)
        
        Debug.Print oWeldBody.GetBody.Name
        Debug.Print oWeldBody.Cutlist.Description

        If i = 0 Then
        
            GetVisibleBodiesList.AddtoList oWeldBody
        
        Else
        
            If False = CheckWhetherTheBodyisWithInAnotherBody(oWeldBody, GetVisibleBodiesList.Items, GetVisibleBodiesList.Count - 1, ParamMin, ParamMax) Then
            
                GetVisibleBodiesList.AddtoList oWeldBody
            
            End If
        
        End If
    
    Next i
    
End Function

Private Function CheckWhetherTheBodyisWithInAnotherBody(WeldBodyToCheck As IWeldBody, vItems As Variant, Idx As Integer, keyParamMin, keyParamMax)

    CheckWhetherTheBodyisWithInAnotherBody = False

    Dim i As Integer
    For i = Idx To LBound(vItems) Step -1
    
        Dim oWeldBody As IWeldBody
        Set oWeldBody = vItems(i)
        
        If Not Left(oWeldBody.Cutlist.Description, 1) = "L" Then

            If CallByName(oWeldBody, keyParamMax, VbGet) < CallByName(WeldBodyToCheck, keyParamMin, VbGet) Then
            
                Exit For
                
            End If
            
            If (WeldBodyToCheck.xMin > oWeldBody.xMin Or Abs(WeldBodyToCheck.xMin - oWeldBody.xMin) <= 0.0001) And _
                (WeldBodyToCheck.xMax < oWeldBody.xMax Or Abs(WeldBodyToCheck.xMax - oWeldBody.xMax) <= 0.0001) And _
                (WeldBodyToCheck.yMin > oWeldBody.yMin Or Abs(WeldBodyToCheck.yMin - oWeldBody.yMin) <= 0.0001) And _
                (WeldBodyToCheck.yMax < oWeldBody.yMax Or Abs(WeldBodyToCheck.yMax - oWeldBody.yMax) <= 0.0001) And _
                (WeldBodyToCheck.zMin > oWeldBody.zMin Or Abs(WeldBodyToCheck.zMin - oWeldBody.zMin) <= 0.0001) And _
                (WeldBodyToCheck.zMax < oWeldBody.zMax Or Abs(WeldBodyToCheck.zMax - oWeldBody.zMax) <= 0.0001) Then
                
                CheckWhetherTheBodyisWithInAnotherBody = True
                Exit For
                
            End If
            
        End If

    Next i

End Function

Sub AddPerimeterBeamProperty(ArrList As IArrListObject, SortParamMin As String, SortParamMax As String, ByRef FirstBeam As IWeldBody, ByRef LastBeam As IWeldBody) 'Sort with Min Parameter

    ArrList.SortItems SortParamMax, True
    Set LastBeam = ArrList.Items(0)
    LastBeam.IsPerimeter = True
    

    ArrList.SortItems SortParamMin, False
    Set FirstBeam = ArrList.Items(0)
    FirstBeam.IsPerimeter = True

End Sub

Sub FindAndAddBeforeSubWeldments(Dict As Scripting.Dictionary, DictIndex As Scripting.Dictionary, ArrList As IArrListObject, Parameter As String)
    
    Dim CheckParameterMin As String
    Dim CheckParameterMax As String
    
    If Left(Parameter, 1) = "x" Then
    
        CheckParameterMin = "yMin"
        CheckParameterMax = "yMax"
        
    Else
        
        CheckParameterMin = "xMin"
        CheckParameterMax = "xMax"
        
    End If
        
    If ArrList.Count > 0 Then
    
        ArrList.SortItems Parameter, False
        
        Dim i As Integer
        Dim vItems As Variant
        vItems = ArrList.Items

        For i = LBound(vItems) To UBound(vItems)
            
            Dim oWeldBody As IWeldBody
            Set oWeldBody = vItems(i)
            
            Dim Index As Integer
            Index = GetIndexWhenKeyValExceedThisValue(DictIndex, CallByName(oWeldBody, Parameter, VbGet))
            
            Dim FirstWeldBody As IWeldBody
            Set FirstWeldBody = GetWeldBodyAttachedBeforeThisBody(oWeldBody, Dict, Index - 1, CheckParameterMin, CheckParameterMax)
            
            'Debug.Print oWeldBody.GetBody.Name
            'Debug.Print FirstWeldBody.GetBody.Name
            
            If Not FirstWeldBody Is Nothing Then
            
                Call FirstWeldBody.AddToSubWeldmentList(True, oWeldBody)
                Set oWeldBody.BeforeBody = FirstWeldBody
                
            End If
            
        Next i

    End If

End Sub

Sub FindAndAddAfterSubWeldments(Dict As Scripting.Dictionary, DictIndex As Scripting.Dictionary, ArrList As IArrListObject, Parameter As String)
    
    Dim CheckParameterMin As String
    Dim CheckParameterMax As String
    
    If Left(Parameter, 1) = "x" Then
    
        CheckParameterMin = "yMin"
        CheckParameterMax = "yMax"
        
    Else
        
        CheckParameterMin = "xMin"
        CheckParameterMax = "xMax"
        
    End If
    
    If ArrList.Count > 0 Then
    
        ArrList.SortItems Parameter, False
        
        Dim i As Integer
        Dim vItems As Variant
        vItems = ArrList.Items

        For i = LBound(vItems) To UBound(vItems)
            
            Dim oWeldBody As IWeldBody
            Set oWeldBody = vItems(i)
            
            Dim Index As Integer
            Index = GetIndexWhenKeyValExceedThisValue(DictIndex, CallByName(oWeldBody, Parameter, VbGet))
            
            Dim SecondWeldBody As IWeldBody
            Set SecondWeldBody = GetWeldBodyAttachedAfterThisBody(oWeldBody, Dict, Index, CheckParameterMin, CheckParameterMax)
            
            If Not SecondWeldBody Is Nothing Then
            
                Call SecondWeldBody.AddToSubWeldmentList(False, oWeldBody)
                Set oWeldBody.AfterBody = SecondWeldBody
                
            End If
            'Debug.Print oWeldBody.GetBody.Name
            'Debug.Print SecondWeldBody.GetBody.Name
            
        Next i

    End If

End Sub

Function GetIndexWhenKeyValExceedThisValue(Dict As Scripting.Dictionary, ValToCheck As Double)


    Dim vKeys As Variant
    vKeys = Dict.Keys
    
    Dim i As Integer
    For i = LBound(vKeys) To UBound(vKeys)
    
        If vKeys(i) > ValToCheck Then
        
            GetIndexWhenKeyValExceedThisValue = i
            Exit For
            
        End If

    Next i

End Function


Function GetWeldBodyAttachedBeforeThisBody(WeldBodyToCheck As IWeldBody, Dict As Scripting.Dictionary, _
    Idx As Integer, CheckParameterMin As String, CheckParameterMax As String) As IWeldBody

    Dim i As Integer
    Dim j As Integer
    
    Dim vKeys As Variant
    vKeys = Dict.Keys
    
    For j = Idx To LBound(vKeys) Step -1
    
        Dim ArrList As IArrListObject
        Set ArrList = Dict.Item(vKeys(j))
    
        Dim vItems As Variant
        vItems = ArrList.Items
    
        For i = LBound(vItems) To UBound(vItems)
    
            Dim oWeldBody As IWeldBody
            Set oWeldBody = vItems(i)
            
            If (CallByName(WeldBodyToCheck, CheckParameterMin, VbGet) > CallByName(oWeldBody, CheckParameterMin, VbGet) Or _
                Abs(CallByName(WeldBodyToCheck, CheckParameterMin, VbGet) - CallByName(oWeldBody, CheckParameterMin, VbGet)) <= 0.0001) And _
                (CallByName(WeldBodyToCheck, CheckParameterMax, VbGet) < CallByName(oWeldBody, CheckParameterMax, VbGet) Or _
                Abs(CallByName(WeldBodyToCheck, CheckParameterMax, VbGet) - CallByName(oWeldBody, CheckParameterMax, VbGet)) <= 0.0001) Then
        
                If (oWeldBody.zMin < WeldBodyToCheck.zMin Or Abs(oWeldBody.zMin - WeldBodyToCheck.zMin) <= 0.0001) And oWeldBody.zMax > WeldBodyToCheck.zMin Then
        
                    Set GetWeldBodyAttachedBeforeThisBody = oWeldBody
                    Exit Function
                
                End If
    
            End If
        
         Next i
         
    Next j

End Function


Function GetWeldBodyAttachedAfterThisBody(WeldBodyToCheck As IWeldBody, Dict As Scripting.Dictionary, _
    Idx As Integer, CheckParameterMin As String, CheckParameterMax As String) As IWeldBody

    Dim i As Integer
    Dim j As Integer
    
    Dim vKeys As Variant
    vKeys = Dict.Keys
    
    For j = Idx To UBound(vKeys)
    
        Dim ArrList As IArrListObject
        Set ArrList = Dict.Item(vKeys(j))
    
        Dim vItems As Variant
        vItems = ArrList.Items
    
        For i = LBound(vItems) To UBound(vItems)
    
            Dim oWeldBody As IWeldBody
            Set oWeldBody = vItems(i)
            
            If (CallByName(WeldBodyToCheck, CheckParameterMin, VbGet) > CallByName(oWeldBody, CheckParameterMin, VbGet) Or _
                Abs(CallByName(WeldBodyToCheck, CheckParameterMin, VbGet) - CallByName(oWeldBody, CheckParameterMin, VbGet)) <= 0.0001) And _
                (CallByName(WeldBodyToCheck, CheckParameterMax, VbGet) < CallByName(oWeldBody, CheckParameterMax, VbGet) Or _
                Abs(CallByName(WeldBodyToCheck, CheckParameterMax, VbGet) - CallByName(oWeldBody, CheckParameterMax, VbGet)) <= 0.0001) Then
        
                If (oWeldBody.zMin < WeldBodyToCheck.zMin Or Abs(oWeldBody.zMin - WeldBodyToCheck.zMin) <= 0.0001) And oWeldBody.zMax > WeldBodyToCheck.zMin Then
        
                    Set GetWeldBodyAttachedAfterThisBody = oWeldBody
                    Exit Function
                
                End If
    
            End If
        
         Next i
         
    Next j

End Function



'
'Private Sub UpdateFrontViewPosition(vComps As Variant, swDrawing As SldWorks.DrawingDoc, swView As SldWorks.View)
'
'    Dim oStartComp As IComp
'    Set oStartComp = vComps(0)
'
'    Dim oEndComp As IComp
'    Set oEndComp = vComps(UBound(vComps))
'
'    Dim CenterX As Double
'    CenterX = (oStartComp.xMin + oEndComp.xMax) / 2
'
'    Dim viewPosition As Variant
'    viewPosition = swView.Position
'
'    viewPosition(0) = viewPosition(0) + (viewPosition(0) - CenterX)
'
'    swView.Position = viewPosition
'
'End Sub
'

'
'Sub CheckandAddLayer(LayName As String, LayerDesc As String, swLayerMgr As SldWorks.LayerMgr)
'
'    Dim vLayNames As Variant
'    vLayNames = swLayerMgr.GetLayerList
'
'    Dim IsLayerExists As Boolean
'
'    Dim i As Integer
'    For i = 0 To UBound(vLayNames)
'
'        If vLayNames(i) = LayName Then
'
'            IsLayerExists = True
'            Exit For
'
'        End If
'
'    Next i
'
'    If Not (IsLayerExists) Then
'
'        swLayerMgr.AddLayer LayName, LayerDesc, 0, swLineStyles_e.swLineDEFAULT, swLineWeights_e.swLW_NONE
'
'        Dim swLayer As SldWorks.Layer
'        Set swLayer = swLayerMgr.GetLayer(LayName)
'
'        swLayer.Style = swLineStyles_e.swLineCENTER
'        swLayer.Width = swLineWeights_e.swLW_THICK5
'
'    End If
'
'End Sub
'


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

Private Function AddStructuralNotes(swDrawing As SldWorks.DrawingDoc, IsSubWeldmentExists As Boolean) As SldWorks.Note
    
    Dim swSheet As SldWorks.Sheet
    Set swSheet = swDrawing.GetCurrentSheet
    
    swDrawing.ActivateSheet swSheet.GetName

    Dim swStructuralNote As SldWorks.Note
    Dim Note As String
    
    Note = "<FONT size=10PTS style=B>NOTES:" & vbCrLf & _
            "<FONT size=8PTS style=R>1. ALL DIMENSIONS TO FIRST FACE OF CONNECTING PLATE (RIGHT SIDE OF BEAM WEB)." & vbCrLf & _
            "ALL BEAMS TO BE BOLTED TO THE FIRST FACE OF THE CONNECTING PLATES." & vbCrLf & _
            "2. CONNECTING PLATES AT " & Chr(34) & "D" & Chr(34) & " WALL ARE DIMENSIONED TO THE FIRST FACE OF CONNECTING PLATE." & vbCrLf & _
              Chr(34) & "D" & Chr(34) & " WALL PERIMETER BEAM TO BE BOLTED TO THE FAR FACE OF THE CONNECTING PLATES." & vbCrLf & _
              "3. BEAM END TO END."

    If IsSubWeldmentExists Then
    
        Note = Note & vbCrLf & "4. REFER NEXT SHEET FOR SUB WELDMENT DETAILS."

    End If

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
                
    'Debug.Print "C:\FBD\COMMON\BLOCKS\" & ProjectNo & " EXTERNAL ELEVATION KEY.SLDBLK"


End Sub

'Private Sub CleanUpActivateAndAddViewLabel(swDrawing As SldWorks.ModelDoc2, swView As SldWorks.View, wallName As String, YPos As Double, _
'    Optional xPos As Double = 0)
'
'    swDrawing.SetUserPreferenceToggle swUserPreferenceToggle_e.swDisplayOrigins, False
'    swDrawing.SetUserPreferenceToggle swUserPreferenceToggle_e.swDisplayPlanes, False
'    swDrawing.SetUserPreferenceToggle swUserPreferenceToggle_e.swDisplayReferencePoints2, False
'    swDrawing.SetUserPreferenceToggle swUserPreferenceToggle_e.swDisplayCurves, False
'    swDrawing.SetUserPreferenceToggle swUserPreferenceToggle_e.swDisplayAllAnnotations, False
'    swDrawing.SetUserPreferenceToggle swUserPreferenceToggle_e.swDisplayCompAnnotations, False
'    swDrawing.SetUserPreferenceToggle swUserPreferenceToggle_e.swDisplayReferencePoints, False
'    swDrawing.SetUserPreferenceToggle swUserPreferenceToggle_e.swDisplayLiveSections, False
'    swDrawing.SetUserPreferenceToggle swUserPreferenceToggle_e.swDisplayLights, False
'
'    swDrawing.ActivateView swView.Name
'
'    Dim SheetDesc As String
'    Dim LabelText As String
'    If InStr(wallName, "Wall") > 0 Then
'
'        SheetDesc = "STRUCTURAL, ELEVATION, EXTERNAL PANELS, " & UCase(wallName)
'        LabelText = "<FONT size=10PTS style=B> $PRP:" & Chr(34) & "SHEET DESCRIPTION" & Chr(34) & _
'         vbCrLf & "<FONT size=8PTS style=R> (INTERIOR VIEW)"
'
'    Else
'
'        SheetDesc = "STRUCTURAL, " & UCase(wallName)
'        LabelText = "<FONT size=10PTS style=B> $PRP:" & Chr(34) & "SHEET DESCRIPTION" & Chr(34)
'
'    End If
'    swDrawing.Extension.CustomPropertyManager("").Set2 "SHEET DESCRIPTION", SheetDesc
'    swDrawing.Extension.CustomPropertyManager("").Set2 "ISSUED FOR", "CONSTRUCTION"
'
'    If xPos = 0 Then
'
'        Dim vOutline As Variant
'        vOutline = swView.GetOutline
'        xPos = (vOutline(0) + vOutline(2)) / 2
'
'    End If
'
'    Dim swLabelNote As SldWorks.Note
'
'    Set swLabelNote = swDrawing.CreateText2(LabelText, xPos, YPos, 0, 0, 0)
'    swLabelNote.SetTextJustification swTextJustification_e.swTextJustificationCenter
'
'    swDrawing.Extension.Rebuild swRebuildOptions_e.swCurrentSheetDisp
'
'End Sub
'



'Sub AddNoteToView(swDrawing As SldWorks.DrawingDoc, NoteText As String, xPos As Double, YPos As Double)
'
'    Dim swNote As SldWorks.Note
'    Set swNote = swDrawing.InsertNote(NoteText)
'
'    If Not swNote Is Nothing Then
'
'        Dim swAnnotation As SldWorks.Annotation
'        Set swAnnotation = swNote.GetAnnotation()
'
'        If Not swAnnotation Is Nothing Then
'
'            swAnnotation.SetPosition xPos, YPos, 0
'
'        End If
'
'    End If
'
'End Sub
'
'Sub CreateRibSketches(ByRef swSketchSegmentHor As SldWorks.SketchSegment, ByRef swSketchSegmentVer As SldWorks.SketchSegment, _
'                    xMin As Double, xMax As Double, yMin As Double, yMax As Double, CompPos As Integer, swSketchMgr As SldWorks.SketchManager, _
'                        OffSetVer As Double, OffSetHor As Double)
'
'    Const Length As Double = 3
'    Const FrontOffset As Double = 1.5
'
'    If CompPos = 0 Then
'
'        Set swSketchSegmentHor = swSketchMgr.CreateLine(xMax - OffSetHor * 0.0254, yMin + FrontOffset * 0.0254, _
'                                0, xMax - (OffSetHor - Length) * 0.0254, yMin + FrontOffset * 0.0254, 0)
'
'        Set swSketchSegmentVer = swSketchMgr.CreateLine(xMin + FrontOffset * 0.0254, yMax - OffSetVer * 0.0254, _
'                                0, xMin + FrontOffset * 0.0254, yMax - (OffSetVer - Length) * 0.0254, 0)
'
'    Else
'
'        Set swSketchSegmentHor = swSketchMgr.CreateLine(xMin + OffSetVer * 0.0254, yMin + FrontOffset * 0.0254, _
'                                0, xMin + (OffSetVer - Length) * 0.0254, yMin + FrontOffset * 0.0254, 0)
'
'        Set swSketchSegmentVer = swSketchMgr.CreateLine(xMax - FrontOffset * 0.0254, yMax - OffSetHor * 0.0254, _
'                                0, xMax - FrontOffset * 0.0254, yMax - (OffSetHor - Length) * 0.0254, 0)
'
'    End If
'
'End Sub
'


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
