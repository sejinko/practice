create table tb_emp_recursive_test (
	employee_id serial primary key,
	full_name  varchar not null,
	manager_id int
	);
	
INSERT INTO TB_EMP_RECURSIVE_TEST (
EMPLOYEE_ID, FULL_NAME, MANAGER_ID)
VALUES
(1 , '�̰��', NULL)
, (2 , '������', 1)
, (3 , '��¹�', 1)
, (4 , '�ϼ���', 1)
, (5 , '�۹鼱', 1)
, (6 , '�̽���', 2)
, (7 , 'ȫ�߼�', 2)
, (8 , '��̼�', 2)
, (9 , '�輱��', 2)
, (10, '�̼���', 3)
, (11, '�輱��', 3)
, (12, '�輱��', 3)
, (13, '�̿���', 3)
, (14, '����', 4)
, (15, '�̽ÿ�', 4)
, (16, '�ּ���', 7)
, (17, '������', 7)
, (18, '�ֹ���', 8)
, (19, '������', 8)
, (20, '�밡��', 8)
;

COMMIT;

select * from TB_EMP_RECURSIVE_TEST;


with recursive tmp1 as (
select 
	employee_id,
	manager_id,
	full_name,
	0 lvl
from
	tb_emp_recursive_test
where manager_id is null 
union 
select 
	e.employee_id,
	e.manager_id,
	e.full_name,
	s.lvl + 1
from
	tb_emp_recursive_test e,
	tmp1 s
where s.employee_id = e.manager_id
)

select 
	employee_id,
	manager_id,
	lpad(' ', 4 * (lvl)) || full_name as full_name
from
	tmp1;


with recursive tmp1 as (
select 
	employee_id,
	manager_id,
	full_name,
	0 lvl
from
	tb_emp_recursive_test 
where employee_id = 2
union 
select
	e.employee_id,
	e.manager_id,
	e.full_name,
	s.lvl + 1
from
	tb_emp_recursive_test e,
	tmp1 s
where s.employee_id = e.manager_id
)
select
	employee_id,
	manager_id
	lpad(' ', 4*(lvl) || full_name as full_name
from
	tmp1;
	