Attribute VB_Name = "ArrangeCoverPlates"
Public Enum CoverPlateSide_e
    Left = 0
    Right = 1
    Top = 2
    Bottom = 3
End Enum

Sub FindAndAddBeforeCoverPlates(Dict As Scripting.Dictionary, ArrList As IArrListObject, Parameter As String, CoverPlateSide As CoverPlateSide_e)
    
    Dim CheckParameterMin As String
    Dim CheckParameterMax As String
'
    If VBA.Left(Parameter, 1) = "x" Then

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
            
            Dim oCoverPlate As IWeldBody
            Set oCoverPlate = vItems(i)
            
            Dim Index As Integer
            Dim IsFound As Boolean
            Index = GetKeyIndexGreaterThanThisVal(Dict, CallByName(oCoverPlate, Parameter, VbGet), IsFound)
            
            If IsFound Then
            
                Dim BeforeCoverPlate As IWeldBody
                Set BeforeCoverPlate = GetCoverPlateBeforeThisCoverPlate(oCoverPlate, Dict, Index - 1, CheckParameterMin, CheckParameterMax)
    
                
                If Not BeforeCoverPlate Is Nothing Then
                
                    Call AddBeforeorAfterCoverPlateProperty(oCoverPlate, BeforeCoverPlate, CoverPlateSide)
    
                End If
                
            End If
            
        Next i

    End If

End Sub

Sub FindAndAddAfterCoverPlates(Dict As Scripting.Dictionary, ArrList As IArrListObject, Parameter As String, CoverPlateSide As CoverPlateSide_e)
    
    Dim CheckParameterMin As String
    Dim CheckParameterMax As String
    
    If VBA.Left(Parameter, 1) = "x" Then
    
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
            
            Dim oCoverPlate As IWeldBody
            Set oCoverPlate = vItems(i)
            
            Dim Index As Integer
            Dim IsFound As Boolean
            Index = GetKeyIndexGreaterThanThisVal(Dict, CallByName(oCoverPlate, Parameter, VbGet), IsFound)
            
            If IsFound Then
            
                Dim AfterCoverPlate As IWeldBody
                Set AfterCoverPlate = GetCoverPlateAfterThisCoverPlate(oCoverPlate, Dict, Index, CheckParameterMin, CheckParameterMax)
                
                If Not AfterCoverPlate Is Nothing Then
                    
                    Call AddBeforeorAfterCoverPlateProperty(oCoverPlate, AfterCoverPlate, CoverPlateSide)
    
                End If
                
            End If

            
        Next i

    End If

End Sub

Function GetKeyIndexGreaterThanThisVal(Dict As Scripting.Dictionary, Val As Double, ByRef IsFound As Boolean) As Integer
    
    If Dict.Count > 0 Then
    
        Dim vKeys As Variant
        vKeys = Dict.Keys

        IsFound = False
        
        Dim i As Integer
        For i = LBound(vKeys) To UBound(vKeys)
            
            If CDbl(vKeys(i)) > Val Then
                
                GetKeyIndexGreaterThanThisVal = i
                IsFound = True
                Exit For
            
            End If
        
        Next i
        
    End If
    
End Function




Function GetCoverPlateBeforeThisCoverPlate(CoverPlateToCheck As IWeldBody, Dict As Scripting.Dictionary, _
    Idx As Integer, CheckMinParam As String, CheckMaxParam As String) As IWeldBody

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
    
            Dim oCoverPlate As IWeldBody
            Set oCoverPlate = vItems(i)
            

            If (((CallByName(oCoverPlate, CheckMinParam, VbGet) < CallByName(CoverPlateToCheck, CheckMinParam, VbGet) Or _
                Abs(CallByName(oCoverPlate, CheckMinParam, VbGet) - CallByName(CoverPlateToCheck, CheckMinParam, VbGet)) <= 0.0001) And _
                CallByName(CoverPlateToCheck, CheckMinParam, VbGet) < CallByName(oCoverPlate, CheckMaxParam, VbGet)) Or _
                ((CallByName(oCoverPlate, CheckMinParam, VbGet) > CallByName(CoverPlateToCheck, CheckMinParam, VbGet) Or _
                Abs(CallByName(oCoverPlate, CheckMinParam, VbGet) - CallByName(CoverPlateToCheck, CheckMinParam, VbGet)) <= 0.0001) And _
                (CallByName(oCoverPlate, CheckMinParam, VbGet) < CallByName(CoverPlateToCheck, CheckMaxParam, VbGet)))) Then
        
                Set GetCoverPlateBeforeThisCoverPlate = oCoverPlate
                Exit Function
    
            End If
        
         Next i
         
    Next j

End Function


Function GetCoverPlateAfterThisCoverPlate(CoverPlateToCheck As IWeldBody, Dict As Scripting.Dictionary, _
    Idx As Integer, CheckMinParam As String, CheckMaxParam As String) As IWeldBody

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
    
            Dim oCoverPlate As IWeldBody
            Set oCoverPlate = vItems(i)
            
             If (((CallByName(oCoverPlate, CheckMinParam, VbGet) < CallByName(CoverPlateToCheck, CheckMinParam, VbGet) Or _
                Abs(CallByName(oCoverPlate, CheckMinParam, VbGet) - CallByName(CoverPlateToCheck, CheckMinParam, VbGet)) <= 0.0001) And _
                CallByName(CoverPlateToCheck, CheckMinParam, VbGet) < CallByName(oCoverPlate, CheckMaxParam, VbGet)) Or _
                ((CallByName(oCoverPlate, CheckMinParam, VbGet) > CallByName(CoverPlateToCheck, CheckMinParam, VbGet) Or _
                Abs(CallByName(oCoverPlate, CheckMinParam, VbGet) - CallByName(CoverPlateToCheck, CheckMinParam, VbGet)) <= 0.0001) And _
                (CallByName(oCoverPlate, CheckMinParam, VbGet) < CallByName(CoverPlateToCheck, CheckMaxParam, VbGet)))) Then
        
        
                Set GetCoverPlateAfterThisCoverPlate = oCoverPlate
                Exit Function

    
            End If
        
         Next i
         
    Next j

End Function

Sub AddBeforeorAfterCoverPlateProperty(oCoverPlate As IWeldBody, PropCoverPlate As IWeldBody, CoverPlateSide As CoverPlateSide_e)
    
    Select Case CoverPlateSide
    
        Case CoverPlateSide_e.Bottom
            
            Set oCoverPlate.BottomWeldBody = PropCoverPlate
        
        Case CoverPlateSide_e.Top
        
            Set oCoverPlate.TopWeldBody = PropCoverPlate
            
        Case CoverPlateSide_e.Left
            
            Set oCoverPlate.LeftWeldBody = PropCoverPlate
            
        Case CoverPlateSide_e.Right
            
            Set oCoverPlate.RightWeldBody = PropCoverPlate
    
    End Select
    
End Sub


