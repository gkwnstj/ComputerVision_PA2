clear all; clc;
% Code to veryfy that it works: 
Q1 = rand(3,5);
Q2 = rand(3,5);
Evec   = calibrated_fivepoint( Q1,Q2);
for i=1:size(Evec,2)   
  E = reshape(Evec(:,i),3,3);
  % Check determinant constraint! 
  det(E)
  % Check trace constraint
  2 *E*transpose(E)*E -trace( E*transpose(E))*E
  % Check reprojection errors
  diag( Q1'*E*Q2)
end