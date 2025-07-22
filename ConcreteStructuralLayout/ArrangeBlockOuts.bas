Attribute VB_Name = "ArrangeBlockOuts"
Public Enum BlockOutSide_e
    Left = 0
    Right = 1
    Top = 2
    Bottom = 3
End Enum

Sub FindAndAddBeforeBlockOuts(Dict As Scripting.Dictionary, ArrList As IArrListObject, Parameter As String, BlockOutSide As BlockOutSide_e)
    
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
            
            Dim oBlockOut As IBlockOut
            Set oBlockOut = vItems(i)
            
            Dim Index As Integer
            Dim IsFound As Boolean
            Index = GetKeyIndexGreaterThanThisVal(Dict, CallByName(oBlockOut, Parameter, VbGet), IsFound)
            
            If IsFound Then
            
                Dim BeforeBlockOut As IBlockOut
                Set BeforeBlockOut = GetBlockOutBeforeThisBlockOut(oBlockOut, Dict, Index - 1, CheckParameterMin, CheckParameterMax)
    
                
                If Not BeforeBlockOut Is Nothing Then
                
                    Call AddBeforeorAfterBlockOutProperty(oBlockOut, BeforeBlockOut, BlockOutSide)
    
                End If
                
            End If
            
        Next i

    End If

End Sub

Sub FindAndAddAfterBlockOuts(Dict As Scripting.Dictionary, ArrList As IArrListObject, Parameter As String, BlockOutSide As BlockOutSide_e)
    
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
            
            Dim oBlockOut As IBlockOut
            Set oBlockOut = vItems(i)
            
            Dim Index As Integer
            Dim IsFound As Boolean
            Index = GetKeyIndexGreaterThanThisVal(Dict, CallByName(oBlockOut, Parameter, VbGet), IsFound)
            
            If IsFound Then
            
                Dim AfterBlockOut As IBlockOut
                Set AfterBlockOut = GetBlockOutAfterThisBlockOut(oBlockOut, Dict, Index, CheckParameterMin, CheckParameterMax)
                
                If Not AfterBlockOut Is Nothing Then
                    
                    Call AddBeforeorAfterBlockOutProperty(oBlockOut, AfterBlockOut, BlockOutSide)
    
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




Function GetBlockOutBeforeThisBlockOut(BlockOutToCheck As IBlockOut, Dict As Scripting.Dictionary, _
    Idx As Integer, CheckMinParam As String, CheckMaxParam As String) As IBlockOut

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
    
            Dim oBlockOut As IBlockOut
            Set oBlockOut = vItems(i)
            

            If (((CallByName(oBlockOut, CheckMinParam, VbGet) < CallByName(BlockOutToCheck, CheckMinParam, VbGet) Or _
                Abs(CallByName(oBlockOut, CheckMinParam, VbGet) - CallByName(BlockOutToCheck, CheckMinParam, VbGet)) <= 0.0001) And _
                CallByName(BlockOutToCheck, CheckMinParam, VbGet) < CallByName(oBlockOut, CheckMaxParam, VbGet)) Or _
                ((CallByName(oBlockOut, CheckMinParam, VbGet) > CallByName(BlockOutToCheck, CheckMinParam, VbGet) Or _
                Abs(CallByName(oBlockOut, CheckMinParam, VbGet) - CallByName(BlockOutToCheck, CheckMinParam, VbGet)) <= 0.0001) And _
                (CallByName(oBlockOut, CheckMinParam, VbGet) < CallByName(BlockOutToCheck, CheckMaxParam, VbGet)))) Then
        
                Set GetBlockOutBeforeThisBlockOut = oBlockOut
                Exit Function
    
            End If
        
         Next i
         
    Next j

End Function


Function GetBlockOutAfterThisBlockOut(BlockOutToCheck As IBlockOut, Dict As Scripting.Dictionary, _
    Idx As Integer, CheckMinParam As String, CheckMaxParam As String) As IBlockOut

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
    
            Dim oBlockOut As IBlockOut
            Set oBlockOut = vItems(i)
            
             If (((CallByName(oBlockOut, CheckMinParam, VbGet) < CallByName(BlockOutToCheck, CheckMinParam, VbGet) Or _
                Abs(CallByName(oBlockOut, CheckMinParam, VbGet) - CallByName(BlockOutToCheck, CheckMinParam, VbGet)) <= 0.0001) And _
                CallByName(BlockOutToCheck, CheckMinParam, VbGet) < CallByName(oBlockOut, CheckMaxParam, VbGet)) Or _
                ((CallByName(oBlockOut, CheckMinParam, VbGet) > CallByName(BlockOutToCheck, CheckMinParam, VbGet) Or _
                Abs(CallByName(oBlockOut, CheckMinParam, VbGet) - CallByName(BlockOutToCheck, CheckMinParam, VbGet)) <= 0.0001) And _
                (CallByName(oBlockOut, CheckMinParam, VbGet) < CallByName(BlockOutToCheck, CheckMaxParam, VbGet)))) Then
        
        
                Set GetBlockOutAfterThisBlockOut = oBlockOut
                Exit Function

    
            End If
        
         Next i
         
    Next j

End Function

Sub AddBeforeorAfterBlockOutProperty(oBlockOut As IBlockOut, PropBlockOut As IBlockOut, BlockOutSide As BlockOutSide_e)
    
    Select Case BlockOutSide
    
        Case BlockOutSide_e.Bottom
            
            Set oBlockOut.BottomBlockOut = PropBlockOut
        
        Case BlockOutSide_e.Top
        
            Set oBlockOut.TopBlockOut = PropBlockOut
            
        Case BlockOutSide_e.Left
            
            Set oBlockOut.LeftBlockOut = PropBlockOut
            
        Case BlockOutSide_e.Right
            
            Set oBlockOut.RightBlockOut = PropBlockOut
    
    End Select
    
End Sub


