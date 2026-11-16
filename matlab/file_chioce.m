function [file_name,bin_target,bin_primary] = file_chioce(number)

numbers = [17,18,19,25,26,30,31,40,54,280,283,310,311,320];
if number<17
    number = numbers(number);
end

switch number
    case 17
        file_name = '19931107_135603_starea.cdf';
        bin_target = {8,9,10,11};
        bin_primary = 9;
        disp('TargetA')
    case 18
        file_name = '19931107_141630_starea.cdf';
        bin_target = {8,9,10,11};
        bin_primary = 9;
        disp('TargetA')
    case 19
        file_name = '19931107_145028_starea.cdf';
        bin_target = {7,8,9};
        bin_primary = 8;
        disp('TargetB')
    case 25
        file_name = '19931108_213827_starea.cdf';
        bin_target = {6,7,8};
        bin_primary = 7;
        disp('TargetA')
    case 26
        file_name = '19931108_220902_starea.cdf';
        bin_target = {6,7,8};
        bin_primary = 7;
        disp('TargetA')
    case 30
        file_name = '19931109_191449_starea.cdf';
        bin_target = {6,7,8};
        bin_primary = 7;
        disp('TargetA')
    case 31
        file_name = '19931109_202217_starea.cdf';
        bin_target = {6,7,8,9};
        bin_primary = 7;
        disp('TargetA')
    case 40
        file_name = '19931110_001635_starea.cdf';
        bin_target = {5,6,7,8};
        bin_primary = 7;
        disp('TargetA')
    case 54
        file_name = '19931111_163625_starea.cdf';
        bin_target = {7,8,9,10};
        bin_primary = 8;
        disp('TargetA')
    case 280
        file_name = '19931118_023604_stareC0000.cdf';
        bin_target = {7,8,9,10};
        bin_primary = 8;
        disp('TargetC')
    case 283
        file_name = '19931118_035737_stareC0000.cdf';
        bin_target = {8,9,10,11,12};
        bin_primary = 10;
        disp('TargetC')
    case 310
        file_name = '19931118_162155_stareC0000.cdf';
        bin_target = {6,7,8,9};
        bin_primary = 7;
        disp('TargetC')
    case 311
        file_name = '19931118_162658_stareC0000.cdf';
        bin_target = {6,7,8,9};
        bin_primary = 7;
        disp('TargetC')
    case 320
        file_name = '19931118_174259_stareC0000.cdf';
        bin_target = {6,7,8,9};
        bin_primary = 7;
        disp('TargetC')
end
disp(file_name)