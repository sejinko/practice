-- ����7��
select * from rental limit 10;

select
	*,
	date(return_date) - date(rental_date) + 1 as sub
from
	rental
where
	date(return_date) - date(rental_date) + 1 >= 10;




  -- datetime() �Լ� ã�ƺ���
select 
	datetime(return_date)
from
	rental;



  -- datediff() �Լ� ã�ƺ���
select 
	*
from
	rental
where 
	datediff(month, return_date, rental_date) > 1;



  -- cast( as date) / datetime ã�ƺ���
select 
	cast(return_date as date)
from
	rental;
	


-- ���� 8
  -- mod(customor_id, 50) : ������
  -- floor(customor_id/50) : ��

select * from customer;
select
	customer_id,
	last_name ||', '|| first_name as full_name,
	email,
	mod(customer_id, 50) as mod_50
from
	customer
where 
	mod(customer_id, 50) = 0;
	


-- ���� 10��
select * from city limit 10;

select 
	count(*)
from 
	city c;

select
	count (distinct city)
from
	city;
	

select
	city
from
	city
group by
	city
having
	count(city) > 1;


  -- �ߺ��� �� ã��
select 
	*
from
	city
where 
	city = 'London'
	

select 
	max(city)
from
	city;
	

select 
	count(distinct city_id)
from
	city;
	

--  ���� 10
select * from actor limit 10;

select
	distinct upper(first_name || ' ' || last_name) as full_name
from
	actor;